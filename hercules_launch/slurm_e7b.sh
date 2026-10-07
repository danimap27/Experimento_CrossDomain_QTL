#!/bin/bash
#SBATCH --job-name=e7b
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=1-20%20
#SBATCH --output=logs/slurm/e7b_%A_%a.out
#SBATCH --error=logs/slurm/e7b_%A_%a.err

# E7b: lambda insurance for synaptic intelligence on the 10-task chain
# (the 5-task calibration may be too strong when the penalty accumulates
# over ten tasks).
cd "$HOME/crossdomain_qcl" || exit 1
export PATH="$HOME/envs/qcl/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cmds_e7b.txt)
[ -z "$ARGS" ] && exit 0
echo "[e7b] $(hostname) $(date) :: $ARGS"
python e234_runner.py $ARGS
