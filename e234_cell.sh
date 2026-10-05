#!/bin/bash
# Run one E2/E3/E4 cell on this node (skips when the result JSON exists).
# Usage: bash e234_cell.sh "--campaign chain4 --arm synth --seed 0 [flags]"
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ARGS="$*"
KEY=$(printf '%s' "$ARGS" | md5sum | cut -c1-12)
mkdir -p logs/e234
LOG="logs/e234/${KEY}.log"
if .venv/bin/python e234_runner.py $ARGS > "$LOG" 2>&1; then
    echo "[done] $ARGS"
else
    echo "[fail] $ARGS (see $LOG)"; exit 1
fi
