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
        sys.exit(0)
with (root / "server.log").open("ab") as log:
    proc = subprocess.Popen(
        [str(root / ".venv/bin/python"), "-u", str(root / "serve.py"), *sys.argv[1:]],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        start_new_session=True,
    )
pid_file.write_text(f"{proc.pid}\n")
print(f"{profile['alias']} starting on {options.host}:{options.port} (PID {proc.pid}). See server.log.")
