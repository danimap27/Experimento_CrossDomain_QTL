#!/bin/bash
#SBATCH --job-name=e9b
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=1-30%15
#SBATCH --output=logs/slurm/e9b_%A_%a.out
#SBATCH --error=logs/slurm/e9b_%A_%a.err

# E9b: comparability cells for the capacity curve.
# q8r  = 8 qubits at the reduced 3000-sample budget (pairs with q12r)
# q12f = 12 qubits at the full 12000-sample budget (pairs with q4/q6/q8).
cd "$HOME/crossdomain_qcl" || exit 1
export PATH="$HOME/envs/qcl/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cmds_e9b.txt)
[ -z "$ARGS" ] && exit 0
echo "[e9b] $(hostname) $(date) :: $ARGS"
python e234_runner.py $ARGS
