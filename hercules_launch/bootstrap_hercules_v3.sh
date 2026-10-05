#!/bin/bash
set -ex
cd "$HOME/crossdomain_qcl" || exit 1
exec > bootstrap3.log 2>&1

wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME/miniconda.sh"
bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda3"
rm -f "$HOME/miniconda.sh"

"$HOME/miniconda3/bin/conda" create -y -p "$HOME/envs/qcl" python=3.11
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
sbatch slurm_e234c.sh
echo "BOOTSTRAP3 DONE $(date)"
