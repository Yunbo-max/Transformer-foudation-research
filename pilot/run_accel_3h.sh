#!/usr/bin/env bash
# Entire project-11 acceleration attempt capped at 3 hours.
# Usage: bash pilot/run_accel_3h.sh [cuda|cpu|mps]
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${1:-cuda}"
case "$DEVICE" in cuda|cpu|mps) ;; *) echo "Usage: $0 [cuda|cpu|mps]" >&2; exit 2;; esac
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU coreutils timeout is required" >&2; exit 2
fi
export WUKE_ACCEL_DEVICE="$DEVICE"
timeout --signal=TERM --kill-after=20s 10800s bash -c '
  set -euo pipefail
  python3 -m unittest discover -s pilot -p "test_*.py"
  python3 pilot/run_accel.py --suite all --device "$WUKE_ACCEL_DEVICE" \
    --out .pilot_workspace_pointer_accel/reports/accel.json
  python3 pilot/run_sparse_model.py --device "$WUKE_ACCEL_DEVICE" \
    --budget-seconds 9600
'
