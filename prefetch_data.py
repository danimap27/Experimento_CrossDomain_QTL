"""Pre-download MNIST, Fashion-MNIST, CIFAR-10 and MobileNetV2 weights into
./data and ./torch_cache so the cluster nodes don't need internet.

Run on the login node (or locally before tar-uploading the project)."""

import os
import torchvision
import torchvision.transforms as T
from torchvision import models

os.environ.setdefault("TORCH_HOME", os.path.abspath("./torch_cache"))
os.makedirs("./data", exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

t = T.ToTensor()
print("MNIST...");          torchvision.datasets.MNIST("./data",        train=True,  download=True, transform=t)
torchvision.datasets.MNIST("./data",        train=False, download=True, transform=t)
print("Fashion-MNIST...");  torchvision.datasets.FashionMNIST("./data", train=True,  download=True, transform=t)
torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=t)
print("CIFAR-10...");       torchvision.datasets.CIFAR10("./data",      train=True,  download=True, transform=t)
torchvision.datasets.CIFAR10("./data",      train=False, download=True, transform=t)
print("MobileNetV2 weights...");  models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
print("Done. TORCH_HOME=", os.environ["TORCH_HOME"])
