#!/usr/bin/env bash
# One complete attempt, including checkout/data prep: <=3h on GNU/Linux.
# status 124 = timed out; inspect the saved partial results, never count as pass.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU coreutils 'timeout' is required for a strict global wall-clock cap." >&2
  exit 2
fi
timeout --signal=TERM --kill-after=20s 10800s python3 pilot/run.py --budget-seconds 10200 "$@"
