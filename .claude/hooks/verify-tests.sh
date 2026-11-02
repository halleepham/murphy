#!/bin/bash
# Stop hook: run the test suite before a completion claim reaches the reader.
#
# Why this exists: in this repo's setup session two workers each wrote files,
# lost track of having done so, and reported them as pre-existing. The code was
# correct; the reports were false. A model's account of its own work is not
# evidence. This check costs no tokens and cannot misremember.
#
# Exit 0 = let the turn end. Exit 2 = block and send the output back as feedback.

set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}" || exit 0

# No test suite yet -> nothing to verify. Stay silent.
# find, not a glob: macOS ships bash 3.2, which has no globstar.
[ -d tests ] || exit 0
[ -n "$(find tests -name 'test_*.py' -type f -print -quit 2>/dev/null)" ] || exit 0

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

if "$PY" -c 'import pytest' 2>/dev/null; then
  out=$("$PY" -m pytest -q 2>&1); rc=$?
else
  # -t . keeps the repo root importable so tests can import the package under
  # test. It needs tests/ to be a package; fall back if it isn't, so a missing
  # __init__.py is never reported as a test failure.
  out=$("$PY" -m unittest discover -s tests -t . 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q 'Start directory is not importable'; then
    out=$("$PY" -m unittest discover -s tests 2>&1); rc=$?
  fi
fi

[ $rc -eq 0 ] && exit 0

# Failing tests: block the turn and hand the output back.
{
  echo "Test suite FAILED (exit $rc). Do not report this work as complete."
  echo "Fix the failures or say plainly that tests are failing and why."
  echo
  printf '%s\n' "$out" | tail -40
} >&2
exit 2
