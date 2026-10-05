#!/bin/bash
# Launch a work list in the background on this node (survives ssh disconnect).
# Usage: bash launch_remote.sh <jobs.txt> <workers>
set -u
cd "$(dirname "$0")"
JOBS="$1"; WORKERS="$2"
LOG="logs/e234_launch_$(hostname).log"
mkdir -p logs/e234
nohup bash e234_launch.sh "$JOBS" "$WORKERS" > "$LOG" 2>&1 < /dev/null &
echo "launched $JOBS with $WORKERS workers, pid $!"
echo "log: $(pwd)/$LOG"
