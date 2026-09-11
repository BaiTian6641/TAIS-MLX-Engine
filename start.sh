#!/bin/sh
set -eu
cd /Users/flare/codex-test-dir/k2-mlx
exec .venv/bin/python start_server.py "$@"
