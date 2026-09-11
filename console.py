"""Interactive model selector for `tais serve`.

Shows every profile with what it costs on this machine right now - the context it
can hold once its weights are resident, and how fast it generates - and starts the
one that is chosen. Arrow keys or j/k to move, enter to serve, r to re-read memory
numbers, q to quit.

Memory is sampled live, so the context column reflects what is actually free: a
browser or another server changes it, and `r` re-reads it without leaving.

Without a terminal on stdin the same table is printed and the selection is read as
a number, so the command works in a script or a pipe.
"""
import contextlib
import os
from pathlib import Path
import shutil
import time
import sys

ROOT = Path(__file__).resolve().parent

RESET = '\033[0m'
DIM = '\033[2m'
BOLD = '\033[1m'
INVERSE = '\033[7m'
GREEN = '\033[32m'
YELLOW = '\033[33m'


def human_tokens(count):
    if not count:
        return '-'
    for limit, suffix in ((1_000_000, 'M'), (1_000, 'k')):
        if count >= limit:
            return f'{count / limit:.0f}{suffix}'
    return str(count)


def human_gib(value):
    return f'{value:.1f}' if value else '-'


def collect(profiles, resolve, memory, costs_for):
    """One row per profile, with everything the selector displays."""
    rows = []
    for alias in profiles:
        row = {'alias': alias, 'available': False, 'size_gib': None, 'kind': '-',
               'max_context': None, 'native_context': None, 'decode': None,
               'prefill': None, 'source': '-'}
        try:
            profile = resolve(alias)
        except Exception as exc:
            row['reason'] = str(exc).split('.')[0][:58]
            rows.append(row)
            continue
        row['available'] = True
        config = profile['config'].get('text_config', profile['config'])
        experts = config.get('num_experts') or config.get('n_routed_experts') or 0
        row['kind'] = f'MoE {experts}' if experts else 'dense'
        row['layers'] = config.get('num_hidden_layers')
        row.update(costs_for(profile))
        rows.append(row)
    return rows


def window(rows, cursor, height):
    """The slice of rows to draw, and how many are hidden above and below.

    A terminal shorter than the profile list used to push the selection off the
    bottom with no way to see it; the view scrolls around the cursor instead.
    """
    overhead = 6           # title, blank, header, blank, footnote, footnote
    visible = max(3, height - overhead)
    if len(rows) <= visible:
        return 0, len(rows), 0, 0
    first = max(0, min(cursor - visible // 2, len(rows) - visible))
    return first, first + visible, first, len(rows) - (first + visible)


# Which columns survive a narrow terminal. The table is rendered by rich, which
# fits it to the width; these decide what to drop rather than squeezing every
# column until none of them can be read.
COLUMN_SETS = {
    'minimal': ('alias', 'max_context', 'decode', 'status'),
    'compact': ('alias', 'weights', 'kind', 'max_context', 'decode', 'status'),
    'full': ('alias', 'weights', 'kind', 'max_context', 'native', 'decode', 'prefill', 'status'),
}


def columns_for(width, mode='auto'):
    if mode in COLUMN_SETS:
        return COLUMN_SETS[mode]
    if width < 80:
        return COLUMN_SETS['minimal']
    if width < 112:
        return COLUMN_SETS['compact']
    return COLUMN_SETS['full']


def cycle_mode(mode, step):
    """Move between the column sets; left shows fewer columns, right more."""
    order = ('minimal', 'compact', 'full')
    current = order.index(mode) if mode in order else         order.index('compact')
    return order[max(0, min(len(order) - 1, current + step))]


def status_cell(row, instances):
    """What the selected profile is doing, from its instance record."""
    running = instances.get(row['alias'])
    if not running:
        return '[dim]stopped[/dim]'
    up = int(time.time() - running.get('started', time.time()))
    bits = [f'[green]:{running["port"]}[/green]', f'{up // 60}m{up % 60:02d}s']
    if running.get('rate'):
        bits.append(f'{running["rate"]:.0f} tok/s')
    if running.get('active') is not None:
        bits.append(f'{running["active"]} req')
    return ' '.join(bits)


def row_cells(row, instances):
    """The value of every column for one profile, as rich markup."""
    if not row['available']:
        blank = {'weights': '-', 'kind': '-', 'max_context': '-', 'native': '-',
                 'decode': '-', 'prefill': '-', 'status': '[yellow]not downloaded[/yellow]'}
        return {'alias': row['alias'], **blank}
    mark = '' if row.get('source') == 'measured' else '[dim]~[/dim]'
    return {
        'alias': row['alias'],
        'weights': human_gib(row.get('weights_gib')),
        'kind': row.get('kind', '-'),
        'max_context': human_tokens(row.get('max_context')),
        'native': human_tokens(row.get('native_context')),
        'decode': f'{row["decode"]:.0f}{mark}' if row.get('decode') else '-',
        'prefill': f'{row["prefill"]:.0f}' if row.get('prefill') else '-',
        'status': status_cell(row, instances),
    }


def render(rows, cursor, memory, width, height=40, instances=None, watching=None,
           notice=None, mode='auto'):
    """The whole screen as a list of lines, none wider than the terminal.

    Rendering is delegated to rich rather than assembled from padded strings.
    A padded f-string counts an ANSI escape as a column and cannot know the
    terminal's width, so a table built that way overflows and the terminal wraps
    it apart - which is exactly what happened here. rich measures the visible
    width, fits the table, and truncates or wraps cells inside their columns.
    """
    from rich.console import Console, Group
    from rich.table import Table
    from rich.text import Text

    instances = instances or {}
    columns = columns_for(width, mode)
    free = memory['available'] / 2**30
    total = memory['total'] / 2**30
    running = sum(1 for alias in instances if instances[alias])

    title = Text.from_markup(
        f'[bold]TAIS MLX Engine[/bold] [dim]services[/dim]   '
        f'[dim]memory[/dim] {free:.0f} of {total:.0f} GiB free   '
        f'[dim]running[/dim] {running}'
        + ('' if mode == 'auto' else f'   [dim]columns[/dim] {mode}'))
    keys = Text.from_markup(
        '[dim]enter start/monitor, s start, x stop, l logs, r refresh, '
        'left/right columns, q quit[/dim]')

    labels = {'alias': 'model', 'weights': 'weights', 'kind': 'kind',
              'max_context': 'ctx (here)', 'native': 'native', 'decode': 'decode',
              'prefill': 'prefill', 'status': 'status'}
    table = Table(box=None, pad_edge=False, show_header=True, padding=(0, 1))
    for name in columns:
        table.add_column(labels[name], no_wrap=(name != 'status'),
                         justify='left' if name in ('alias', 'kind', 'status') else 'right',
                         overflow='ellipsis' if name == 'alias' else 'crop')

    first, last, above, below = window(rows, cursor, height)
    if above:
        table.add_row(*[''] * len(columns))
    for index in range(first, last):
        cells = row_cells(rows[index], instances)
        style = 'reverse' if index == cursor else None
        marker = '> ' if index == cursor else '  '
        row_values = []
        for position, name in enumerate(columns):
            value = cells[name]
            row_values.append((marker + value) if position == 0 else value)
        table.add_row(*row_values, style=style)
    if below:
        table.add_row(*[''] * len(columns))

    footer = [Text.from_markup(
        '[dim]ctx (here) is what fits beside the weights in current free memory; '
        'native is the model\'s declared window. ~ marks an estimate, not a '
        'measurement.[/dim]')]
    if watching:
        footer.append(Text(''))
        footer.extend(Text.from_markup(line) for line in watching)
    if notice:
        footer.append(Text(''))
        footer.append(Text.from_markup(f'[yellow]{notice}[/yellow]'))

    console_ = Console(width=width, force_terminal=False, highlight=False,
                       no_color=os.environ.get('NO_COLOR') is not None, soft_wrap=False)
    with console_.capture() as captured:
        console_.print(Group(title, Text(''), table, Text(''), *footer))
    # rich fits every rendered line to the console width; splitting its output is
    # therefore guaranteed to give lines the terminal can display unwrapped.
    return captured.get().rstrip('\n').split('\n')


# The byte sequences terminals send, mapped to actions. Kept as data rather than
# inline so every key a user can press is enumerable and testable.
ESCAPE_ACTIONS = {
    b'[A': 'up', b'[B': 'down', b'[C': 'right', b'[D': 'left',
    b'OA': 'up', b'OB': 'down', b'OC': 'right', b'OD': 'left',
    b'[5~': 'pageup', b'[6~': 'pagedown',
    b'[H': 'home', b'[F': 'end', b'[1~': 'home', b'[4~': 'end',
}


def escape_action(tail):
    """What an escape sequence means: an arrow, a jump, or 'other'.

    A bare escape is 'escape' (quit). Anything unrecognised is 'other' and is
    ignored, so a function key or a Home key never quits the selector.
    """
    if tail == b'':
        return 'escape'
    return ESCAPE_ACTIONS.get(tail, 'other')


@contextlib.contextmanager
def raw_terminal(fd):
    """Raw mode for the whole interactive session, restored on every exit path.

    Setting it per keystroke and restoring it between reads - which is what this
    did first - leaves the terminal in cooked mode whenever the user is not
    actively being read from, so an arrow key pressed at the wrong moment is
    buffered by the line discipline and never arrives as a key.
    """
    import termios
    import tty

    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def enter_raw(fd):
    """Put the terminal in raw mode and return what it takes to restore it."""
    import termios
    import tty

    saved = termios.tcgetattr(fd)
    tty.setraw(fd)
    return saved


def restore_terminal(fd, saved):
    import termios

    termios.tcsetattr(fd, termios.TCSADRAIN, saved)


# One byte may be read too many when a key sequence ends and the next begins in
# the same burst; it is pushed back here so read_key sees it rather than losing it.
_PUSHBACK = bytearray()


def read_byte(fd, timeout=None):
    """One byte, honouring anything pushed back, or None if none arrives."""
    if _PUSHBACK:
        byte = bytes(_PUSHBACK[:1])
        del _PUSHBACK[:1]
        return byte
    if timeout is not None and not select_ready(fd, timeout):
        return None
    return os.read(fd, 1)


def read_escape_tail(fd, budget=0.06):
    """Everything a terminal sends after ESC, matched as it arrives.

    A terminal sends an arrow as one write but the reader may see it arrive in
    pieces, so the tail is accumulated until it is a known sequence, cannot become
    one, or the budget runs out. Whatever is read is consumed either way, so an
    unrecognised key never leaks into the next read.
    """
    tail = b''
    deadline = time.monotonic() + budget
    while len(tail) < 8:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        part = read_byte(fd, max(0.0, remaining))
        if not part:
            break
        if part == b'\x1b' and tail:
            # The next key's escape, arriving in the same burst: keep it for the
            # caller instead of consuming it into this sequence.
            _PUSHBACK.extend(part)
            break
        tail += part
        if tail in ESCAPE_ACTIONS:
            break
        if not any(sequence.startswith(tail) for sequence in ESCAPE_ACTIONS):
            # Not one we act on, but it is still a control sequence: consume it to
            # its final byte so its parameters cannot be read as the next key.
            while len(tail) < 16:
                following = read_byte(fd, 0.02)
                if not following:
                    break
                if 0x40 <= following[0] <= 0x7e:
                    break
                tail += following
            break
    return tail


def read_key(fd, timeout=None):
    """One keypress: an arrow key, a character, None on end of input.

    The terminal must already be raw - the interactive views hold it that way for
    the session. With a timeout, ``timeout`` comes back instead of blocking, which
    is what lets the service view tick while nothing is pressed.
    """
    char = read_byte(fd, timeout) if timeout is not None else read_byte(fd)
    if char is None:
        return 'timeout'
    if char == b'':
        return None
    if char in (b'\x03', b'\x1c'):      # ctrl-c, ctrl-\: raw mode has no signals
        return 'escape'
    if char == b'\x04':                 # ctrl-d
        return None
    if char == b'\x1b':
        return escape_action(read_escape_tail(fd))
    if char in (b'\r', b'\n'):
        return 'enter'
    return char.decode('utf-8', 'ignore')


def select_ready(fd, timeout=0.02):
    import select as _select
    ready, _, _ = _select.select([fd], [], [], timeout)
    return bool(ready)


def choose(rows, memory, costs_for, interactive=True, refresh=None, recompute=None):
    """Return the chosen row, or None if the user quit."""
    terminal = shutil.get_terminal_size((100, 30))
    width, height = terminal.columns, terminal.lines
    cursor = next((i for i, row in enumerate(rows) if row['available']), 0)
    mode = 'auto'

    if not interactive:
        for index, row in enumerate(rows, 1):
            state = row.get('source', '-') if row['available'] else 'not downloaded'
            decode = f'{row["decode"]:.1f}' if row.get('decode') else '-'
            prefill = f'{row["prefill"]:.0f}' if row.get('prefill') else '-'
            print(f'  [{index:>2}] {row["alias"]:<22} '
                  f'{human_gib(row.get("weights_gib")):>7} GiB  '
                  f'ctx {human_tokens(row.get("max_context")):>7}  '
                  f'decode {decode:>7}  prefill {prefill:>7}  {state}'
                  f'{" " + row.get("reason", "") if row.get("reason") else ""}')
        try:
            answer = input('serve which model? [number or name, empty to quit] ').strip()
        except EOFError:
            return None
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(rows):
            return rows[int(answer) - 1]
        return next((row for row in rows if row['alias'] == answer), None)

    fd = sys.stdin.fileno()
    print('\033[?25l', end='')  # hide the cursor
    try:
        with raw_terminal(fd):
            while True:
                terminal = shutil.get_terminal_size((100, 30))
                width, height = terminal.columns, terminal.lines
                sys.stdout.write('\033[2J\033[H')
                sys.stdout.write('\n'.join(render(rows, cursor, memory, width, height,
                                                  mode=mode)) + '\n')
                sys.stdout.flush()
                key = read_key(fd)
                if key is None or key in ('q', 'escape'):
                    return None
                if key == 'other':
                    continue
                if key in ('up', 'k'):
                    cursor = (cursor - 1) % len(rows)
                elif key in ('down', 'j'):
                    cursor = (cursor + 1) % len(rows)
                elif key in ('left', 'right'):
                    mode = cycle_mode(mode, +1 if key == 'right' else -1)
                elif key == 'pageup':
                    cursor = max(0, cursor - max(1, height - 8))
                elif key == 'pagedown':
                    cursor = min(len(rows) - 1, cursor + max(1, height - 8))
                elif key == 'home':
                    cursor = 0
                elif key == 'end':
                    cursor = len(rows) - 1
                elif key == 'r':
                    if refresh:
                        memory = refresh()
                    if recompute is not None:
                        rows[:] = recompute()
                elif key == 'enter':
                    return rows[cursor]
    finally:
        print('\033[?25h', end='')


def monitor_lines(record, log_tail=6):
    """Live panel for one instance: its sample, then the last log lines."""
    import services

    port = record['port']
    sample = services.metrics(port) or {}
    extra = sample.get('extra') or {}
    lines = [f'[bold]{record.get("alias")}[/bold] [dim]on port {port}, pid {record.get("pid")}[/dim]']
    for key in ('decode_tps', 'prefill_tps', 'requests', 'active', 'completed', 'errors'):
        value = sample.get(key, extra.get(key))
        if value is not None:
            lines.append(f'  [dim]{key}[/dim] {value}')
    for key in ('mlx_active_gib', 'mlx_peak_gib', 'context_limit', 'disk_entries'):
        if extra.get(key) is not None:
            lines.append(f'  [dim]{key}[/dim] {extra[key]}')
    tail = services.tail(port, log_tail)
    if tail:
        lines.append('  [dim]log[/dim]')
        for entry in tail:
            lines.append(f'  [dim]{entry[-100:].replace("[", "\\[")}[/dim]')
    return lines


def free_port(start=8080):
    """The first port nothing is listening on and no record claims."""
    import services
    import socket

    taken = {record['port'] for record in services.instances(include_dead=True)}
    port = start
    while port < start + 40:
        if port not in taken:
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', port)) != 0:
                    return port
        port += 1
    return start


def services_view(rows, memory, costs_for, refresh=None, recompute=None):
    """Start, monitor and stop instances from one screen."""
    import services

    terminal = shutil.get_terminal_size((100, 40))
    width, height = terminal.columns, terminal.lines
    cursor = next((i for i, row in enumerate(rows) if row['available']), 0)
    notice = None
    watching = None
    mode = 'auto' 

    def live():
        found = {}
        for record in services.instances():
            sample = services.metrics(record['port']) or {}
            extra = sample.get('extra') or {}
            found[record['alias']] = {
                **record,
                'rate': extra.get('decode_tps') or sample.get('decode_tps'),
                'active': sample.get('active', sample.get('in_flight')),
            }
        return found

    instances = live()
    fd = sys.stdin.fileno()
    print('\033[?25l', end='')
    saved_terminal = enter_raw(fd)
    try:
        while True:
            terminal = shutil.get_terminal_size((100, 40))
            width, height = terminal.columns, terminal.lines
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.write('\n'.join(render(rows, cursor, memory, width, height, instances,
                                              watching, notice, mode)) + '\n')
            sys.stdout.flush()
            key = read_key(fd, timeout=1.0)
            if key == 'timeout':
                instances = live()
                if watching:
                    running = instances.get(rows[cursor]['alias'])
                    watching = monitor_lines(running) if running else None
                continue
            if key is None or key in ('q', 'escape'):
                return 0
            if key in ('other',):
                continue
            if key in ('up', 'k'):
                cursor = (cursor - 1) % len(rows)
                watching = None
            elif key in ('down', 'j'):
                cursor = (cursor + 1) % len(rows)
                watching = None
            elif key in ('left', 'right'):
                mode = cycle_mode(mode, +1 if key == 'right' else -1)
            elif key in ('pageup', 'pagedown', 'home', 'end'):
                step = max(1, height - 8)
                cursor = {'pageup': max(0, cursor - step),
                          'pagedown': min(len(rows) - 1, cursor + step),
                          'home': 0, 'end': len(rows) - 1}[key]
                watching = None
            elif key == 'r':
                if refresh:
                    memory = refresh()
                if recompute is not None:
                    rows[:] = recompute()
                instances = live()
                notice = None
            elif key in ('enter', 's'):
                row = rows[cursor]
                if not row['available']:
                    notice = f'{row["alias"]} is not downloaded: tais download {row["alias"]}'
                    continue
                running = instances.get(row['alias'])
                if running:
                    watching = monitor_lines(running)
                    notice = None
                    continue
                port = free_port()
                notice = f'starting {row["alias"]} on port {port}...'
                sys.stdout.write('\033[2J\033[H' + notice + '\n')
                sys.stdout.flush()
                record = services.start(row['alias'], port=port)
                notice = (f'{row["alias"]} listening on {record["port"]} (pid {record["pid"]})'
                          if record.get('alive') else f'start failed: {record.get("error")}')
                instances = live()
            elif key == 'x':
                row = rows[cursor]
                running = instances.get(row['alias'])
                if not running:
                    notice = f'{row["alias"]} is not running'
                    continue
                result = services.stop(running['port'])
                notice = (f'stopped {row["alias"]}' if result.get('stopped')
                          else f'not stopped: {result.get("reason")}')
                watching = None
                instances = live()
            elif key in ('l', 'm'):
                row = rows[cursor]
                running = instances.get(row['alias'])
                if not running:
                    notice = f'{row["alias"]} is not running'
                    continue
                watching = None if key == 'l' and watching else monitor_lines(running)
                if key == 'l' and watching is None:
                    watching = monitor_lines(running, log_tail=14)
                notice = None
    finally:
        restore_terminal(fd, saved_terminal)
        print('\033[?25h', end='')


def main(argv=None):
    """Entry point for the service console, and for `tais pick` with --pick."""
    import argparse

    import hf_env
    from model_profiles import PROFILES, parse_options, resolve_profile
    from profile_costs import profile_costs
    from runtime_support import memory_info

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--print', action='store_true', dest='dry_run',
                        help='print the table and the command, serve nothing')
    parser.add_argument('--pick', action='store_true',
                        help='choose one model and serve it in the foreground, '
                             'instead of managing instances')
    hf_env.add_arguments(parser)
    args = parser.parse_args(argv or [])
    hf_env.apply_from_args(args, root=ROOT)

    import mlx.core as mx

    def snapshot():
        info = mx.device_info()
        return memory_info(), info['max_recommended_working_set_size']

    def costs_for(profile):
        memory, working_set = snapshot()
        return profile_costs(profile, memory['available'], working_set,
                             resident_bytes=mx.get_active_memory())

    def resolve(alias):
        profile = resolve_profile(parse_options(['--model', alias]))
        profile['alias'] = alias
        return profile

    memory, _ = snapshot()
    rows = collect(PROFILES, resolve, memory, costs_for)
    recompute = lambda: collect(PROFILES, resolve, snapshot()[0], costs_for)  # noqa: E731

    if not args.pick and not args.dry_run and sys.stdin.isatty():
        # The default screen manages instances: start, watch and stop them.
        return services_view(rows, memory, costs_for,
                             refresh=lambda: snapshot()[0], recompute=recompute)

    chosen = choose(rows, memory, costs_for, interactive=sys.stdin.isatty(),
                    refresh=lambda: snapshot()[0], recompute=recompute)
    if chosen is None:
        return 0
    if not chosen['available']:
        print(f'{chosen["alias"]} is not downloaded yet: tais download {chosen["alias"]}',
              file=sys.stderr)
        return 1
    if args.dry_run:
        print(f'tais serve --model {chosen["alias"]}')
        return 0

    # Serve in the foreground: the selector owns the terminal until then, and
    # replacing the process hands it over cleanly.
    python = str(ROOT / '.venv' / 'bin' / 'python')
    if not Path(python).exists():
        python = sys.executable
    print(f'serving {chosen["alias"]}; ctrl-c to stop')
    os.execv(python, [python, str(ROOT / 'serve.py'), '--model', chosen['alias']])


if __name__ == '__main__':
    sys.exit(main())
