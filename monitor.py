"""Live local console: .venv/bin/python monitor.py [--once]."""
import argparse
import json
from pathlib import Path
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent


def percent(value):
    if value is None:
        return 'unavailable'
    width = max(0, min(20, round(value / 5)))
    return f'{"━" * width}{"─" * (20-width)} {value:5.1f}%'


def gib(value):
    return f'{value/2**30:.2f} GiB'


def render():
    try:
        data = json.loads((ROOT / 'metrics.json').read_text())
    except (OSError, ValueError):
        return Panel('No server metrics yet. Start with: sh start.sh', title='K2 Horizon')
    stale = time.time() - data['updated'] > 4
    status = '[red]OFFLINE / stale snapshot[/red]' if stale else '[green]LIVE[/green]'
    overview = (f'{status}  PID {data["pid"]}  |  Running {data["running"]}/{data["concurrency"]}'
                f'  Queued {data["queued"]}  Oldest wait {data["queue_oldest_seconds"]:.1f}s\n'
                f'Generation: {data["tokens_per_second"]:.1f} tok/s (last 5s)  |  '
                f'Completed {data["completed"]}  Failed {data["failed"]}')
    hardware = Table.grid(padding=(0, 2))
    hardware.add_column(); hardware.add_column()
    hardware.add_row('CPU · whole system', percent(data.get('cpu_percent')))
    hardware.add_row('GPU · whole device', percent(data.get('gpu_percent')))
    hardware.add_row('Server CPU · 100% per core', f'{data.get("process_cpu_percent", 0):.1f}%')
    mem = data.get('memory', {})
    if mem:
        hardware.add_row('Unified RAM · system', f'{gib(mem["used"])} used / {gib(mem["total"])}  ({gib(mem["available"])} available estimate)')
    hardware.add_row('MLX memory', f'{gib(data.get("mlx_active_bytes",0))} active  |  {gib(data.get("mlx_pool_bytes",0))} allocator pool  |  {gib(data.get("mlx_peak_bytes",0))} peak')
    hardware.add_row('Context · last admission', f'{data.get("context_limit", "loading")} tokens  /  {data.get("native_limit", "?")} model maximum')
    hardware.add_row('Inactive KV · SSD', f'{data.get("disk_entries",0)} caches  {gib(data.get("disk_bytes",0))}  |  hits {data.get("cache_hits",0)}  misses {data.get("cache_misses",0)}')
    table = Table('Request', 'State', 'Wait', 'Prefill', 'Output', 'Decode tok/s', 'TTFT', expand=True)
    now = data['updated']
    for row in data['requests'] + data['recent']:
        start = row.get('started', now)
        first = row.get('first_token')
        end = row.get('decode_end', row.get('finished', now))
        speed = (row['tokens']-1)/max(.001,end-first) if first and row['tokens']>1 else 0
        table.add_row(row['id'], row['phase'], f'{start-row["submitted"]:.1f}s',
                      f'{row["prompt_done"]}/{row["prompt_total"]}', str(row['tokens']),
                      f'{speed:.1f}', f'{first-row["submitted"]:.1f}s' if first else '—')
    if 'expert_budget_bytes' in data:
        hits, misses = data.get('expert_hits',0), data.get('expert_misses',0)
        hardware.add_row('MoE hot experts', f'{gib(data.get("expert_cache_bytes",0))} / {gib(data["expert_budget_bytes"])}  |  {100*hits/max(1,hits+misses):.1f}% hits')
        hardware.add_row('MoE SSD reads', f'{gib(data.get("expert_read_bytes",0))} cumulative  |  {data.get("expert_evictions",0)} evictions')
    if 'embedding_budget_bytes' in data:
        hits, misses = data.get('embedding_hits',0), data.get('embedding_misses',0)
        hardware.add_row('N-gram row cache', f'{gib(data.get("embedding_cache_bytes",0))} / {gib(data["embedding_budget_bytes"])}  |  {100*hits/max(1,hits+misses):.1f}% hits')
        hardware.add_row('N-gram SSD reads', gib(data.get('embedding_read_bytes',0)) + ' cumulative')
    parts = [Panel(overview, title=f'{data.get("model", "K2 Horizon")} · live console'), Panel(hardware, title='Hardware & cache'), table]
    if data.get('cache_error'):
        parts.append(Text('SSD cache: ' + data['cache_error'], style='yellow'))
    if data.get('sampler_error'):
        parts.append(Text('Sampling: ' + data['sampler_error'], style='yellow'))
    parts.append(Text('Ctrl-C closes monitor. Server continues running.', style='dim'))
    return Group(*parts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.once:
        Console().print(render())
    else:
        try:
            with Live(render(), refresh_per_second=2, screen=True) as live:
                while True:
                    time.sleep(.5)
                    live.update(render())
        except KeyboardInterrupt:
            pass
