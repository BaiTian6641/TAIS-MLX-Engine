"""Detach the server from the launching terminal/process group."""
import os
from pathlib import Path
import subprocess
import sys
from model_profiles import parse_options, resolve_profile

root = Path(__file__).resolve().parent
options = parse_options(sys.argv[1:])
profile = resolve_profile(options)
pid_file = root / "server.pid"
if pid_file.exists():
    try:
        pid = int(pid_file.read_text())
        os.kill(pid, 0)
    except (ValueError, ProcessLookupError):
        pass
    else:
        print(f"Server is already running (PID {pid}).")
        sys.exit(1)

# A pid file only tells us about servers *this* launcher started. A stale server
# without one - from a previous session, or spawned some other way - would make
# the new one fail to bind, and then every test silently talks to the old one,
# which is exactly the trap this exists to catch.
import socket as _socket

with _socket.socket() as _probe:
    if _probe.connect_ex(('127.0.0.1', options.port)) == 0:
        print(f"Port {options.port} already has a server listening on it - either "
              f"an old instance or something else. `tais status` shows ours; "
              f"stop it or pass --port.")
        sys.exit(1)
with (root / "server.log").open("ab") as log:
    proc = subprocess.Popen(
        [str(root / ".venv/bin/python"), "-u", str(root / "serve.py"), *sys.argv[1:]],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        start_new_session=True,
    )
pid_file.write_text(f"{proc.pid}\n")
print(f"{profile['alias']} starting on {options.host}:{options.port} (PID {proc.pid}). See server.log.")
