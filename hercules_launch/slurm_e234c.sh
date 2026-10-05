#!/bin/bash
#SBATCH --job-name=e234
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=1-290%48
#SBATCH --output=logs/slurm/e234_%A_%a.out
#SBATCH --error=logs/slurm/e234_%A_%a.err

cd "$HOME/crossdomain_qcl" || exit 1
export PATH="$HOME/envs/qcl/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cmds_e234.txt)
[ -z "$ARGS" ] && exit 0
echo "[e234] $(hostname) $(date) :: $ARGS"
python e234_runner.py $ARGS
