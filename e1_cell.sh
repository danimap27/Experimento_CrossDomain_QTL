#!/bin/bash
# Run one E1 cell (profile seed) unless its manifest already exists.
# Usage: bash e1_cell.sh <profile> <seed>
set -u
PROFILE="$1"; SEED="$2"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")"
mkdir -p logs/e1
MAN="results/e1_manifest__${PROFILE}__s${SEED}.json"
if [ -f "$MAN" ]; then
    echo "[skip] ${PROFILE} ${SEED}"
    exit 0
fi
LOG="logs/e1/${PROFILE}_s${SEED}.log"
if .venv/bin/python e1_runner.py --profile "$PROFILE" --seed "$SEED" > "$LOG" 2>&1; then
    echo "[done] ${PROFILE} ${SEED}"
else
    echo "[fail] ${PROFILE} ${SEED} (see $LOG)"
    exit 1
fi
