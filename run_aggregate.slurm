#!/bin/bash
#SBATCH --job-name=crossdomain_agg
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmarper2@upo.es

set -euo pipefail
module load Miniconda3 || true
source activate qtl || source activate qml || true

python3 aggregate_results.py \
    --pattern 'results/results_seed_*.json' \
    --out results_noisy_5seeds.json \
    --out-dir .
