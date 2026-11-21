#!/usr/bin/env bash
# End-to-end bounded pilot: three hours at most, no remote/background work.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU coreutils timeout is required for hard 3h wall-clock enforcement." >&2
  exit 2
fi
timeout --signal=TERM --kill-after=20s 10800s python3 pilot/run.py --budget-seconds 10200 "$@"
