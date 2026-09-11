"""Running instances: start, list, stop and observe them.

One process serves one model, so several can run at once on different ports. Each
instance is recorded in `run/<port>.json` beside its metrics and log, which is
what makes a list view possible at all - the engine's own `server.pid` only ever
described one server.

Every signal goes through a check that the pid really is a server of ours:
records outlive crashes, and a recycled pid must never be signalled.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run'


def record_path(port):
    return RUN / f'{port}.json'


def metrics_path(port):
    return RUN / f'{port}.metrics.json'


def log_path(port):
    return RUN / f'{port}.log'


def python():
    interpreter = ROOT / '.venv' / 'bin' / 'python'
    return str(interpreter) if interpreter.exists() else 'python3'


def process_command(pid):
    try:
        out = subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                             capture_output=True, text=True, timeout=3)
    except Exception:
        return ''
    return out.stdout.strip()


def is_our_server(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    command = process_command(pid)
    return 'serve.py' in command and str(ROOT) in command


def load(port):
    path = record_path(port)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    record['port'] = int(record.get('port', port))
    record['alive'] = is_our_server(record.get('pid'))
    return record


def instances(include_dead=False):
    """Every recorded instance, newest first, with its liveness resolved."""
    if not RUN.exists():
        return []
    found = []
    for path in sorted(RUN.glob('*.json')):
        if path.name.endswith('.metrics.json'):
            continue
        try:
            port = int(path.stem)
        except ValueError:
            continue
        record = load(port)
        if record and (record['alive'] or include_dead):
            found.append(record)
    return sorted(found, key=lambda r: r.get('started', 0), reverse=True)


def forget(port):
    record_path(port).unlink(missing_ok=True)
    metrics_path(port).unlink(missing_ok=True)
    legacy = ROOT / 'server.pid'
    if legacy.exists() and legacy.read_text().strip() == str((load(port) or {}).get('pid', '')):
        legacy.unlink(missing_ok=True)


def start(alias, port=8080, host='0.0.0.0', extra=()):
    """Launch a detached server and wait until it answers.

    Returns the record, with an `error` key if it never became ready.
    """
    RUN.mkdir(exist_ok=True)
    existing = load(port)
    if existing and existing['alive']:
        return {**existing, 'error': f'port {port} already serves {existing.get("alias")}'}

    argv = [python(), '-u', str(ROOT / 'serve.py'), '--model', alias,
            '--host', host, '--port', str(port), *extra]
    with log_path(port).open('ab') as log:
        process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    record = {'pid': process.pid, 'alias': alias, 'port': port, 'host': host,
              'started': time.time(), 'argv': extra}
    record_path(port).write_text(json.dumps(record, indent=2) + '\n')
    if port == 8080:
        (ROOT / 'server.pid').write_text(f'{process.pid}\n')

    deadline = time.time() + 240
    while time.time() < deadline:
        if process.poll() is not None:
            return {**record, 'alive': False, 'error': 'the server exited during start-up; '
                                                       f'see {log_path(port).name}'}
        try:
            import urllib.request
            urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=2).read()
            return {**record, 'alive': True}
        except Exception:
            time.sleep(2)
    return {**record, 'alive': False, 'error': f'not ready within 240s; see {log_path(port).name}'}


def stop(port, timeout=10.0):
    """Signal the instance and wait for it to go, verifying the pid first."""
    record = load(port)
    if record is None:
        return {'stopped': False, 'reason': 'no record for this port'}
    pid = record.get('pid')
    if not is_our_server(pid):
        forget(port)
        return {'stopped': False, 'reason': f'pid {pid} is not one of our servers'}
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_our_server(pid):
            forget(port)
            return {'stopped': True, 'pid': pid, 'alias': record.get('alias')}
        time.sleep(0.2)
    os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)
    forget(port)
    return {'stopped': True, 'pid': pid, 'alias': record.get('alias'), 'forced': True}


def metrics(port):
    """The live sample the server's telemetry writes, if it is running."""
    path = metrics_path(port)
    legacy = ROOT / 'metrics.json'
    if not path.exists() and port == 8080 and legacy.exists():
        path = legacy
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def tail(port, lines=40):
    path = log_path(port)
    legacy = ROOT / 'server.log'
    if not path.exists() and port == 8080 and legacy.exists():
        path = legacy
    if not path.exists():
        return []
    with path.open('rb') as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 64 * 1024))
        return handle.read().decode('utf-8', 'replace').splitlines()[-lines:]


def summary(port):
    """One line worth of state for a running instance."""
    record = load(port) or {}
    sample = metrics(port) or {}
    extra = sample.get('extra') or {}
    active = sample.get('active') or sample.get('in_flight')
    rate = extra.get('decode_tps') or sample.get('decode_tps')
    parts = [f'{record.get("alias", "?")} on port {record.get("port", port)}']
    if record.get('started'):
        parts.append(f'up {int(time.time() - record["started"])}s')
    if rate:
        parts.append(f'{float(rate):.1f} tok/s')
    if active is not None:
        parts.append(f'{active} active')
    if extra.get('mlx_active_gib'):
        parts.append(f'{float(extra["mlx_active_gib"]):.1f} GiB')
    return ' | '.join(parts)
