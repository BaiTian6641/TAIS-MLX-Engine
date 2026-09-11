"""Thread-safe request counters and a local, atomic dashboard snapshot."""
from collections import deque
import json
import os
from pathlib import Path
import resource
import threading
import time
import uuid

from runtime_support import memory_info, cpu_ticks, gpu_usage


class Telemetry:
    def __init__(self):
        self.lock = threading.RLock()
        self.requests = {}
        self.recent = deque(maxlen=8)
        self.events = deque(maxlen=4096)
        self.completed = self.failed = 0
        self.extra = {}
        self.started = time.time()
        self.extra_source = None

    def submit(self, request):
        key = uuid.uuid4().hex[:8]
        request._monitor_id = key
        with self.lock:
            self.requests[key] = {'id': key, 'phase': 'queued', 'submitted': time.time(),
                                  'tokens': 0, 'prompt_done': 0, 'prompt_total': 0}
        return key

    def update(self, key, **values):
        with self.lock:
            if key in self.requests:
                self.requests[key].update(values)

    def token(self, key):
        now = time.time()
        with self.lock:
            row = self.requests[key]
            row.setdefault('first_token', now)
            row['tokens'] += 1
            row['phase'] = 'generating'
            self.events.append(now)

    def finish(self, key, error=False):
        with self.lock:
            row = self.requests.pop(key, None)
            if row is None:
                return
            row['phase'] = 'failed' if error else 'done'
            row['finished'] = time.time()
            self.recent.appendleft(row)
            self.failed += int(error)
            self.completed += int(not error)

    def snapshot(self):
        now = time.time()
        with self.lock:
            rows = [dict(r) for r in self.requests.values()]
            queued = [r for r in rows if r['phase'] == 'queued']
            return dict(self.extra, pid=os.getpid(), updated=now, started=self.started,
                        running=len(rows) - len(queued), queued=len(queued), concurrency=1,
                        queue_oldest_seconds=max((now-r['submitted'] for r in queued), default=0),
                        tokens_per_second=sum(t > now-5 for t in self.events) / 5,
                        completed=self.completed, failed=self.failed,
                        requests=rows, recent=list(self.recent))

    def start_sampler(self, path, mx):
        def sample():
            previous_cpu = None
            previous_process = None
            while True:
                now = time.monotonic()
                extra = {}
                try:
                    ticks = cpu_ticks()
                    if previous_cpu:
                        diffs = [(a-b) % 2**32 for a,b in zip(ticks, previous_cpu)]
                        extra['cpu_percent'] = 100 * (1-diffs[2]/max(1,sum(diffs)))
                    previous_cpu = ticks
                    extra['memory'] = memory_info()
                    extra['gpu_percent'] = gpu_usage()
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    process = usage.ru_utime + usage.ru_stime
                    if previous_process:
                        extra['process_cpu_percent'] = 100*(process-previous_process[0])/(now-previous_process[1])
                    previous_process = process, now
                    extra['mlx_active_bytes'] = mx.get_active_memory()
                    extra['mlx_pool_bytes'] = mx.get_cache_memory()
                    extra['mlx_peak_bytes'] = mx.get_peak_memory()
                    if self.extra_source is not None:
                        extra.update(self.extra_source())
                except Exception as exc:
                    extra['sampler_error'] = str(exc)
                with self.lock:
                    self.extra.update(extra)
                temp = Path(str(path) + '.tmp')
                try:
                    temp.write_text(json.dumps(self.snapshot()))
                    temp.replace(path)
                except OSError:
                    pass
                time.sleep(1)
        threading.Thread(target=sample, daemon=True, name='dashboard-sampler').start()
