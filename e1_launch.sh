#!/bin/bash
# E1 campaign launcher: work list of "<profile> <seed>" lines, N parallel cells.
# Usage: bash e1_launch.sh <jobs.txt> <workers>
set -u
JOBS="$1"; WORKERS="${2:-6}"
cd "$(dirname "$0")"
xargs -P "$WORKERS" -n 2 bash e1_cell.sh < "$JOBS"
echo "[campaign] work list $JOBS finished at $(date)"
