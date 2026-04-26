import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import numpy as np

class DataModule:
    """
    Data module to handle Synthetic datasets, MNIST, Fashion-MNIST, and CIFAR-10 (via MobileNetV2).
    Filters binary classes, standardizes, applies PCA reduction to 4 features,
    and returns PyTorch DataLoaders.
    """
    def __init__(self, data_dir='./data', batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        os.makedirs(data_dir, exist_ok=True)
        # Load backbone once if needed
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.mobilenet.eval()
        for param in self.mobilenet.parameters():
            param.requires_grad = False
            
    def get_mobilenet_features_task(self, classes=(0, 1), n_samples=1000):
        """Loads CIFAR-10, passes through MobileNetV2, then PCA to 4 features."""
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_set = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=transform)
        
        def extract_features(dataset, limit):
            features, labels = [], []
            count = 0
            with torch.no_grad():
                for img, label in dataset:
                    if label in classes:
                        out = self.mobilenet(img.unsqueeze(0)).squeeze()
                        features.append(out.numpy())
                        labels.append(classes.index(label))
                        count += 1
                        if count >= limit: break
            return np.array(features), np.array(labels)
            
        print("Extracting MobileNetV2 features for CIFAR-10...")
        X_tr, y_tr = extract_features(train_set, n_samples)
        X_ts, y_ts = extract_features(test_set, int(n_samples*0.2))
        
        pca = PCA(n_components=4)
        X_tr_pca = pca.fit_transform(X_tr)
        X_ts_pca = pca.transform(X_ts)
        
        # Normalize to [0, pi]
        X_tr_pca = (X_tr_pca - X_tr_pca.min(axis=0)) / (X_tr_pca.max(axis=0) - X_tr_pca.min(axis=0) + 1e-8) * np.pi
        X_ts_pca = (X_ts_pca - X_ts_pca.min(axis=0)) / (X_ts_pca.max(axis=0) - X_ts_pca.min(axis=0) + 1e-8) * np.pi
        
        train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_pca, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)), batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_ts_pca, dtype=torch.float32), torch.tensor(y_ts, dtype=torch.long)), batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def get_synthetic_task(self, n_samples=2000):
        """Generates a synthetic dataset for the Source Domain."""
        X, y = make_classification(
            n_samples=n_samples, n_features=4, n_informative=4, 
            n_redundant=0, n_classes=2, random_state=42
        )
        # Normalize to [0, pi] for Angle Embedding
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * np.pi
        
        train_size = int(0.8 * n_samples)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
            batch_size=self.batch_size, shuffle=False
        )
        return train_loader, test_loader

    def _process_dataset(self, dataset, classes, pca=None, is_train=True, limit_samples=200):
        """Filters a base dataset by classes, limits size for fast QML simulation, and applies PCA."""
        X, y = [], []
        count = 0
        for img, label in dataset:
            if label in classes:
                X.append(img.numpy().flatten())
                y.append(classes.index(label))
                count += 1
                if count >= limit_samples:
                    break
        
        X = np.array(X)
        y = np.array(y)

        if is_train and pca is None:
            pca = PCA(n_components=4)
            X_pca = pca.fit_transform(X)
        else:
            X_pca = pca.transform(X)
        
        # Normalize to [0, pi] for Angle Embedding
        X_pca = (X_pca - X_pca.min(axis=0)) / (X_pca.max(axis=0) - X_pca.min(axis=0) + 1e-8) * np.pi

        return torch.tensor(X_pca, dtype=torch.float32), torch.tensor(y, dtype=torch.long), pca

    def get_mnist_task(self, classes=(0, 1), pca_model=None):
        """Loads MNIST filtered to 2 classes."""
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True, transform=transform)

        X_train, y_train, trained_pca = self._process_dataset(train_set, classes, pca=pca_model, is_train=(pca_model is None), limit_samples=1000)
        X_test, y_test, _ = self._process_dataset(test_set, classes, pca=trained_pca, is_train=False, limit_samples=200)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader, trained_pca

    def get_fashion_mnist_task(self, classes=(0, 1), pca_model=None):
        """Loads Fashion-MNIST filtered to 2 classes."""
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.FashionMNIST(root=self.data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root=self.data_dir, train=False, download=True, transform=transform)

        X_train, y_train, trained_pca = self._process_dataset(train_set, classes, pca=pca_model, is_train=(pca_model is None), limit_samples=1000)
        X_test, y_test, _ = self._process_dataset(test_set, classes, pca=trained_pca, is_train=False, limit_samples=200)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader, trained_pca
