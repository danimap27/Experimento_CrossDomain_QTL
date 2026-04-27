#!/bin/bash
# Sync project (incl. pre-cached data + MobileNet weights) to Hercules.
# Usage:
#   HERCULES_USER=dmarper2 HERCULES_HOST=hercules.spc.cica.es bash deploy.sh
#   HERCULES_USER=dmarper2 HERCULES_HOST=... REMOTE_DIR='~/scratch/crossdomain' bash deploy.sh

set -euo pipefail

HERCULES_USER="${HERCULES_USER:?Set HERCULES_USER env var}"
HERCULES_HOST="${HERCULES_HOST:-hercules.spc.cica.es}"
REMOTE_DIR="${REMOTE_DIR:-~/scratch/Experimento_CrossDomain_QTL}"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  From : $LOCAL_DIR"
echo "  To   : ${HERCULES_USER}@${HERCULES_HOST}:${REMOTE_DIR}"
echo "============================================================"

ssh "${HERCULES_USER}@${HERCULES_HOST}" "mkdir -p ${REMOTE_DIR}"

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'figure*.png' \
    --exclude 'paper_crossdomain_qtl.*' \
    --exclude 'paper.txt' \
    --exclude 'references.bib' \
    --exclude 'run_noisy_5seeds.log' \
    --exclude 'results/*.json' \
    --exclude 'logs' \
    "$LOCAL_DIR/" \
    "${HERCULES_USER}@${HERCULES_HOST}:${REMOTE_DIR}/"

cat <<EOF

Sync done. Next:
  ssh ${HERCULES_USER}@${HERCULES_HOST}
  cd ${REMOTE_DIR}
  module load Miniconda3
  conda env list   # check if 'qtl' or 'qml' env exists
  # If not:
  #   conda create -n qtl python=3.11 -y
  #   source activate qtl
  #   pip install -r requirements.txt
  sbatch run_array.slurm
  # Note JOBID, then:
  sbatch --dependency=afterok:<JOBID> run_aggregate.slurm
  squeue -u \$USER
EOF
