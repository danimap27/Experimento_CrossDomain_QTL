#!/bin/bash
#SBATCH --job-name=e8
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=1-90%32
#SBATCH --output=logs/slurm/e8_%A_%a.out
#SBATCH --error=logs/slurm/e8_%A_%a.err

# E8 study: input capacity vs the hard class-IL benchmarks.
# Qubits/components swept to 6 and 8 (smnist10) and 8 (smnist5) for the
# arms that work (er, derpp) plus the plain baseline.
cd "$HOME/crossdomain_qcl" || exit 1
export PATH="$HOME/envs/qcl/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cmds_e8.txt)
[ -z "$ARGS" ] && exit 0
echo "[e8] $(hostname) $(date) :: $ARGS"
python e234_runner.py $ARGS
