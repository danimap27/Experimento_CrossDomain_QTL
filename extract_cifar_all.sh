#!/bin/bash
# Extract + cache MobileNetV2 features for the 5 CIFAR-10 class pairs in parallel.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p logs/e234
pids=()
for pair in "0 1" "2 3" "4 5" "6 7" "8 9"; do
    name=$(echo "$pair" | tr ' ' '_')
    .venv/bin/python e234_runner.py --extract-cifar --classes $pair \
        --limit-train 2000 --limit-test 400 > "logs/e234/extract_${name}.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
echo "[extract] CIFAR pairs finished rc=$rc at $(date)"
