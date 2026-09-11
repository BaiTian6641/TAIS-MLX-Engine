"""k2mlx command line: serve, download, inspect and measure the engine.

The heavy lifting lives in the modules this dispatches to - ``serve.py`` runs the
HTTP server, ``setup_models.py`` and ``setup_gguf.py`` fetch weights, and the
``check_*`` scripts measure a running server. This file adds the things a
deployment needs around them: one command surface, a preflight check, and Hub
configuration, so a mirror or a token is set once rather than in every script.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

BENCHMARKS = {
    'api': ('check_profile_api.py', 'chat, streaming and exact-prefix continuation'),
    'concurrency': ('check_concurrency.py', 'answers must not change under batching'),
    'context': ('check_context_scaling.py', 'decode and prefill against context length'),
    'optimizations': ('check_optimizations.py', 'prefill, prompt reuse, batching'),
    'speculative': ('check_speculative.py', 'draft acceptance and speedup'),
    'diffusion': ('check_diffusion.py', 'block-diffusion sampler'),
    'runtime': ('check_runtime.py', 'resident memory and device behaviour'),
    'capacity': ('check_capacity.py', 'context admission under a memory budget'),
    'gguf-parity': ('check_gguf_parity.py', 'quantized weights against the reference'),
}


def python():
    venv = ROOT / '.venv' / 'bin' / 'python'
    return str(venv) if venv.exists() else sys.executable


def run(script, argv):
    """Replace this process with one of the engine's scripts."""
    target = ROOT / script
    if not target.exists():
        raise SystemExit(f'{script} is missing from {ROOT}')
    os.execv(python(), [python(), str(target), *argv])


def spawned(argv):
    return subprocess.run([python(), *argv], cwd=ROOT).returncode


def read_pid():
    try:
        return int((ROOT / 'server.pid').read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cmd_serve(args):
    forwarded = list(args.engine_args)
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    if args.detach:
        return spawned([str(ROOT / 'start_server.py'), *forwarded])
    run('serve.py', forwarded)


def cmd_stop(args):
    pid = read_pid()
    if not alive(pid):
        (ROOT / 'server.pid').unlink(missing_ok=True)
        print('no server is running')
        return 0
    os.kill(pid, signal.SIGTERM)
    for _ in range(int(args.timeout * 10)):
        if not alive(pid):
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
    (ROOT / 'server.pid').unlink(missing_ok=True)
    print(f'stopped pid {pid}')
    return 0


def cmd_models(args):
    from model_profiles import PROFILES, parse_options, resolve_profile

    rows = []
    for alias in PROFILES:
        entry = {'model': alias, 'model_type': PROFILES[alias]['model_type'], 'path': PROFILES[alias]['path']}
        try:
            profile = resolve_profile(parse_options(['--model', alias]))
            config = profile['config'].get('text_config', profile['config'])
            entry.update(available=True, layers=config.get('num_hidden_layers'),
                         experts=config.get('num_experts') or config.get('n_routed_experts') or 0,
                         context=config.get('max_position_embeddings'))
            path = profile['path']
            entry['size_gib'] = round(sum(p.stat().st_size for p in path.glob('*.safetensors')) / 2**30, 1)
        except Exception as exc:
            entry.update(available=False, reason=str(exc).split('.')[0])
        rows.append(entry)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    width = max(len(row['model']) for row in rows)
    for row in rows:
        if row['available']:
            kind = 'MoE' if row['experts'] else 'dense'
            print(f"  {row['model']:<{width}}  {row['size_gib']:6.1f} GiB  {row['layers']:>3}L {kind:<5} "
                  f"ctx {row['context']:>7,}  {row['model_type']}")
        else:
            reason = row.get('reason', 'not downloaded').strip()
            hint = 'partial' if 'incomplete' in reason.lower() else 'missing'
            print(f"  {row['model']:<{width}}  {'-':>13}  {hint}: {reason[:60]}"
                  f"  ->  k2mlx download {row['model']}")
    return 0


def cmd_download(args):
    argv = [*args.models]
    if args.metadata_only:
        argv.append('--metadata-only')
    argv += args.hf_args
    return spawned([str(ROOT / 'setup_models.py'), *argv])


def cmd_context(args):
    import context_extension
    from model_profiles import PROFILES

    if args.path:
        path = args.path
    elif args.model in PROFILES:
        path = ROOT / PROFILES[args.model]['path']
    else:
        raise SystemExit('name a profile with --model, or a directory with --path')

    if args.restore:
        print(json.dumps(context_extension.restore(path), indent=2))
        return 0
    result = context_extension.extend(path, args.factor, original=args.original,
                                      max_position=args.max_position, force=args.force,
                                      dry_run=args.dry_run)
    print(json.dumps({**result, 'path': str(path)}, indent=2))
    if result.get('changed'):
        print(f"\n{path.name}: context extended to {result['new_max']:,} tokens "
              f"(YaRN factor {args.factor} on {result['original_max']:,}). "
              f"Restore with `k2mlx context --restore --path {path}`.")
    return 0


def cmd_doctor(args):
    import platform

    import hf_env
    from version import __version__

    problems, notes = [], []
    print(f'k2mlx {__version__}')
    print(f'  python      {platform.python_version()} ({platform.machine()})')

    lock = {}
    for line in (ROOT / 'requirements.lock').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            name, _, pinned = line.partition('==')
            lock[name.strip()] = pinned.split()[0]

    from importlib.metadata import PackageNotFoundError, version as package_version
    for distribution, display in (('mlx', 'mlx'), ('mlx-lm', 'mlx-lm'),
                                  ('transformers', 'transformers'), ('numpy', 'numpy'),
                                  ('safetensors', 'safetensors')):
        try:
            installed = package_version(distribution)
        except PackageNotFoundError as exc:
            problems.append(f'{display} is not installed: {exc}')
            continue
        pinned = lock.get(display) or lock.get(distribution)
        if pinned and installed != pinned and not pinned.startswith(installed):
            notes.append(f'{display} {installed} differs from the pinned {pinned}')
        print(f'  {display:<13} {installed}')

    if platform.system() == 'Darwin':
        total = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
        print(f'  memory      {total / 2**30:.0f} GiB unified')
        free = shutil.disk_usage(ROOT).free
        print(f'  disk free   {free / 2**30:.0f} GiB')
        if free < 50 * 2**30:
            notes.append('less than 50 GiB free; large checkpoints need room to download')
    else:
        notes.append('not running on Apple silicon; MLX requires macOS on arm64')

    configuration = hf_env.configure(root=ROOT)
    print(f"  hub         endpoint {configuration['endpoint']}, token {configuration['token']}, "
          f"cache {configuration['home']}")

    try:
        from model_profiles import PROFILES, parse_options, resolve_profile
        available = [alias for alias in PROFILES
                     if _resolves(alias, resolve_profile, parse_options)]
        print(f'  profiles    {len(available)} of {len(PROFILES)} downloaded')
        if not available:
            notes.append('no profile weights are present; run k2mlx download <model>')
    except Exception as exc:
        problems.append(f'profile table unusable: {exc}')

    pid = read_pid()
    print(f'  server      {"pid " + str(pid) if alive(pid) else "not running"}')

    scratch = ROOT / '.doctor-write-test'
    try:
        scratch.write_text('ok')
        scratch.unlink()
    except OSError as exc:
        problems.append(f'repository root is not writable: {exc}')

    for note in notes:
        print(f'  note: {note}')
    for problem in problems:
        print(f'  problem: {problem}', file=sys.stderr)
    return 1 if problems else 0


def _resolves(alias, resolve_profile, parse_options):
    try:
        resolve_profile(parse_options(['--model', alias]))
        return True
    except Exception:
        return False


def cmd_bench(args):
    script, description = BENCHMARKS[args.kind]
    argv = ['--model', args.model, '--port', str(args.port)]
    if args.kind == 'speculative':
        argv = ['--drafter', args.drafter] if args.drafter else []
        argv += ['--tokens', str(args.tokens)]
    if args.kind == 'optimizations':
        argv += ['--label', args.label or 'bench', '--concurrency', str(args.concurrency)]
    if args.kind == 'concurrency':
        argv += ['--concurrency', str(args.concurrency)]
    if args.kind == 'context':
        argv += ['--lengths', args.lengths] if args.lengths else []
    print(f'{script}: {description}')
    return spawned([str(ROOT / script), *argv])


def cmd_version(args):
    from version import __version__
    print(f'k2mlx {__version__}')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog='k2mlx',
        description='MLX inference engine for Apple silicon: dense, MoE, block-diffusion '
                    'and streaming-expert models.',
        epilog=f'scripts live in {ROOT}; engine options after `serve` are passed through '
               'unchanged (see `k2mlx serve --model <name> --help`).')
    from version import __version__
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    sub = parser.add_subparsers(dest='command', required=True)

    serve = sub.add_parser('serve', help='run the inference server', add_help=False,
                           usage='k2mlx serve [--detach] <engine options>',
                           description='Everything after `serve` is passed to the engine '
                                       'unchanged, except --detach which this command consumes. '
                                       'Run `python serve.py --help` for the engine options.')
    serve.add_argument('--detach', action='store_true',
                       help='start in the background and write server.pid')
    serve.add_argument('engine_args', nargs=argparse.REMAINDER)
    serve.set_defaults(func=cmd_serve)

    stop = sub.add_parser('stop', help='stop a detached server')
    stop.add_argument('--timeout', type=float, default=10.0)
    stop.set_defaults(func=cmd_stop)

    models = sub.add_parser('models', help='list profiles and whether their weights are present')
    models.add_argument('--json', action='store_true')
    models.set_defaults(func=cmd_models)

    import hf_env
    download = sub.add_parser('download', help='fetch profile weights from the Hub or a mirror')
    download.add_argument('models', nargs='+', help='profile names, as shown by `k2mlx models`')
    download.add_argument('--metadata-only', action='store_true',
                          help='config and tokenizer only, no weight shards')
    hf_env.add_arguments(download)
    download.set_defaults(func=cmd_download)

    context = sub.add_parser('context', help='extend a checkpoint context with YaRN rope scaling')
    context.add_argument('--model', help='profile to extend')
    context.add_argument('--path', type=Path, help='checkpoint directory, instead of a profile')
    context.add_argument('--factor', type=float, default=2.0,
                         help='length multiplier over the trained window (default: 2)')
    context.add_argument('--original', type=int, help='trained window, if the config does not say')
    context.add_argument('--max-position', type=int, dest='max_position',
                         help='declared maximum after extension (default: original x factor)')
    context.add_argument('--force', action='store_true', help='replace existing rope scaling')
    context.add_argument('--dry-run', action='store_true', dest='dry_run')
    context.add_argument('--restore', action='store_true', help='put back the saved config')
    context.set_defaults(func=cmd_context)

    doctor = sub.add_parser('doctor', help='check the environment before serving')
    doctor.set_defaults(func=cmd_doctor)

    bench = sub.add_parser('bench', help='measure a running server or a drafter')
    bench.add_argument('kind', choices=sorted(BENCHMARKS))
    bench.add_argument('--model', required=True)
    bench.add_argument('--port', type=int, default=8081)
    bench.add_argument('--drafter', help='drafter directory, for `bench speculative`')
    bench.add_argument('--tokens', type=int, default=64)
    bench.add_argument('--concurrency', type=int, default=1)
    bench.add_argument('--label', help='label for `bench optimizations`')
    bench.add_argument('--lengths', help='comma-separated context lengths for `bench context`')
    bench.set_defaults(func=cmd_bench)

    version = sub.add_parser('version', help='print the engine version')
    version.set_defaults(func=cmd_version)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if 'serve' in argv:
        # `serve` forwards unknown options to the engine, so it is split by hand:
        # argparse's REMAINDER stops recognising them once a flag comes first.
        index = argv.index('serve')
        head, tail = argv[:index], argv[index + 1:]
        args = build_parser().parse_args([*head, 'serve'])
        args.detach = '--detach' in tail
        args.engine_args = [item for item in tail if item != '--detach']
        return _run(args)
    return _run(build_parser().parse_args(argv))


def _run(args):
    if args.command == 'download':
        args.hf_args = []
        for flag, value in (('--hf-endpoint', args.hf_endpoint), ('--hf-token', args.hf_token),
                            ('--hf-home', args.hf_home)):
            if value:
                args.hf_args += [flag, value]
        if args.hf_offline:
            args.hf_args.append('--hf-offline')
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        # `k2mlx models | head` closes the pipe early; say nothing about it.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == '__main__':
    sys.exit(main())
