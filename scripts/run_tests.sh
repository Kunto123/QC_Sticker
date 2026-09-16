#!/usr/bin/env sh
# Run the backend regression suite.
#
# One command that (a) ensures dev deps are importable, (b) runs pytest against
# backend/tests from the repo root so the root conftest.py + pyproject pytest
# config are picked up.
#
# POSIX sh, works in Windows git-bash. Usage:
#   scripts/run_tests.sh                # run whole suite
#   scripts/run_tests.sh -k contract    # pass extra args straight to pytest
set -eu

# Resolve repo root from this script's location (works regardless of CWD).
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# Pick a python. Prefer an active venv's python, else python3, else python.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  PY="${VIRTUAL_ENV}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

echo "[run_tests] repo root : $REPO_ROOT"
echo "[run_tests] python     : $($PY --version 2>&1)"

# (a) Ensure pytest (dev dep) is importable; install on demand if missing.
# pytest is a test-only dep and intentionally NOT in pyproject runtime deps.
if ! "$PY" -c "import pytest" >/dev/null 2>&1; then
  echo "[run_tests] pytest not found — installing pytest..."
  "$PY" -m pip install --quiet pytest
fi

# (b) Run the suite. Extra CLI args ($@) are forwarded to pytest.
echo "[run_tests] running: $PY -m pytest backend/tests -q $*"
exec "$PY" -m pytest backend/tests -q "$@"
