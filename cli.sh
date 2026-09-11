#!/bin/sh
# TAIS MLX Engine launcher.
#
# Runs the console from wherever it is invoked, using the project's virtual
# environment if one exists and the system python otherwise, so a checkout works
# before `pip install -e .` has been run.
#
#   ./cli.sh                 service console
#   ./cli.sh status          what is running
#   ./cli.sh --help          every command
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python="$root/.venv/bin/python"

if [ ! -x "$python" ]; then
    if command -v python3 >/dev/null 2>&1; then
        python=$(command -v python3)
        echo "cli.sh: no virtual environment at $root/.venv, using $python" >&2
    else
        echo "cli.sh: python3 is not on PATH and $root/.venv does not exist" >&2
        echo "        create one: python3 -m venv .venv && .venv/bin/pip install -r requirements.lock" >&2
        exit 1
    fi
fi

# The console reads single keystrokes, which needs a terminal; the entry point
# falls back to printing the table when stdin is not one.
exec "$python" "$root/cli.py" "$@"
