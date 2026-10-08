#!/bin/bash
#SBATCH --job-name=e9
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=1-15%15
#SBATCH --output=logs/slurm/e9_%A_%a.out
#SBATCH --error=logs/slurm/e9_%A_%a.err

# E9: 12-qubit point of the capacity curve (reduced budget: the simulator
# cost grows ~2^n per step; q16/q32 are out of practical scope).
cd "$HOME/crossdomain_qcl" || exit 1
export PATH="$HOME/envs/qcl/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cmds_e9.txt)
[ -z "$ARGS" ] && exit 0
echo "[e9] $(hostname) $(date) :: $ARGS"
python e234_runner.py $ARGS
