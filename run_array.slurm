#!/bin/bash
#SBATCH --job-name=crossdomain_qtl
#SBATCH --partition=standard
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/crossdomain_%A_%a.out
#SBATCH --error=logs/crossdomain_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmarper2@upo.es

# ----------------------------------------------------------------------
# Cross-Domain QTL — 1 seed per array task (Heron r2 noise model).
# Submit:   sbatch run_array.slurm
# After all 5 finish:
#           sbatch --dependency=afterok:<JOBID> run_aggregate.slurm
# or simply: python3 aggregate_results.py
# ----------------------------------------------------------------------

set -euo pipefail
mkdir -p logs results

module load Miniconda3 || true
# Adjust the env name if you use a different one:
source activate qtl || source activate qml || {
    echo "[error] Conda env 'qtl' or 'qml' not found. Create one with:"
    echo "        conda create -n qtl python=3.11 -y"
    echo "        source activate qtl"
    echo "        pip install -r requirements.txt"
    exit 1
}

SEED=${SLURM_ARRAY_TASK_ID}
OUT="results/results_seed_${SEED}.json"

echo "==> Host:    $(hostname)"
echo "==> Date:    $(date)"
echo "==> Seed:    ${SEED}"
echo "==> Out:     ${OUT}"
echo "==> Python:  $(which python3)  ($(python3 --version))"

# Pre-cached datasets/weights live next to the project to avoid network on
# compute nodes. Run prefetch_data.py once on the login node beforehand.
export TORCH_HOME="${SLURM_SUBMIT_DIR}/torch_cache"
echo "==> TORCH_HOME=${TORCH_HOME}"

python3 -u main.py --seed "${SEED}" --out "${OUT}" --noise --epochs 10 --data-dir ./data

echo "==> done seed ${SEED}"
