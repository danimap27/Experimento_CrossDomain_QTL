#!/bin/bash
#SBATCH --job-name=e6
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=1-100%48
#SBATCH --output=logs/slurm/e6_%A_%a.out
#SBATCH --error=logs/slurm/e6_%A_%a.err

# E6 study: continual-learning mechanism grid on the standard 5-task
# benchmark (split-MNIST class-IL, global labels). New arms:
# si / l2 / derpp (scratch) and synth_si / synth_l2 / synth_derpp (prior).
cd "$HOME/crossdomain_qcl" || exit 1
export PATH="$HOME/envs/qcl/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cmds_e6.txt)
[ -z "$ARGS" ] && exit 0
echo "[e6] $(hostname) $(date) :: $ARGS"
python e234_runner.py $ARGS
