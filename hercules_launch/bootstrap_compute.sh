#!/bin/bash
#SBATCH --job-name=bootstrap
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/home/dmartin/crossdomain_qcl/bootstrap5_%j.out
#SBATCH --error=/lustre/home/dmartin/crossdomain_qcl/bootstrap5_%j.err

# Bootstrap E2/E3/E4 v5 — conda-forge (evita el ToS de los canales de Anaconda).
set -ex
cd "$HOME/crossdomain_qcl"
echo "== host: $(hostname) | $(date) =="

if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  rm -rf "$HOME/miniconda3"
  bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda3"
fi
"$HOME/miniconda3/bin/conda" --version

"$HOME/miniconda3/bin/conda" create -y -p "$HOME/envs/qcl" python=3.11 \
  --override-channels --channel conda-forge
"$HOME/envs/qcl/bin/python" --version

"$HOME/envs/qcl/bin/pip" install "pennylane==0.44.1" torch torchvision numpy scikit-learn matplotlib

"$HOME/envs/qcl/bin/python" - <<'PYEOF'
from torchvision import datasets, transforms
tf = transforms.ToTensor()
for name in ("MNIST", "FashionMNIST", "KMNIST"):
    getattr(datasets, name)(root="./data", train=True, download=True, transform=tf)
    getattr(datasets, name)(root="./data", train=False, download=True, transform=tf)
datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
print("DATA PREFETCH OK")
PYEOF

mkdir -p logs/slurm
echo "=== launching campaign array ==="
sbatch slurm_e234c.sh
echo "BOOTSTRAP5 DONE $(date)"
