#!/bin/bash
#SBATCH --job-name=e234g
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=1-110%48
#SBATCH --output=logs/slurm/e234g_%A_%a.out
#SBATCH --error=logs/slurm/e234g_%A_%a.err

# E2 re-run with global labels (true task-IL / class-IL with the shared
# multi-class head): chain4 + smnist5 + sfmnist5, 110 cells. The previous
# local-label results of these campaigns are preserved in results_oldlabels/.
cd "$HOME/crossdomain_qcl" || exit 1
export PATH="$HOME/envs/qcl/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cmds_e234_global.txt)
[ -z "$ARGS" ] && exit 0
echo "[e234g] $(hostname) $(date) :: $ARGS"
python e234_runner.py $ARGS --force
