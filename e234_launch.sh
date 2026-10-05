#!/bin/bash
# E2/E3/E4 campaign launcher: work list of full argument strings, N parallel cells.
# Usage: bash e234_launch.sh <jobs.txt> <workers>
set -u
JOBS="$1"; WORKERS="${2:-5}"
cd "$(dirname "$0")"
xargs -P "$WORKERS" -I{} bash e234_cell.sh {} < "$JOBS"
echo "[campaign] work list $JOBS finished at $(date)"
