#!/bin/bash
#SBATCH --job-name=e7
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=1-140%48
#SBATCH --output=logs/slurm/e7_%A_%a.out
#SBATCH --error=logs/slurm/e7_%A_%a.err

# E7 study: long sequences (10 tasks x 1 class, split-MNIST and split-FMNIST)
# plus hyperparameter tuning on the 5-task benchmark (ewc lam 1e2, er buffer 50%).
cd "$HOME/crossdomain_qcl" || exit 1
export PATH="$HOME/envs/qcl/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cmds_e7.txt)
[ -z "$ARGS" ] && exit 0
echo "[e7] $(hostname) $(date) :: $ARGS"
python e234_runner.py $ARGS
