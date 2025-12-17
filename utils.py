"""Utility functions for data loading and preprocessing."""
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np


class MaskedDataset(torch.utils.data.Dataset):
    """Wrap a dataset and apply a post-transform (e.g., feature mask) to images."""
    def __init__(self, base_dataset, post_transform=None):
        self.base = base_dataset
        self.post_transform = post_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        if self.post_transform is not None:
            image = self.post_transform(image)
        return image, label


class FeatureMaskTransform:
    """Zero out features (pixels) outside the provided mask (C,H,W). Applied after Normalize."""
    def __init__(self, mask: torch.Tensor):
        # Expect mask of shape (C,H,W) with 0/1 values
        self.mask = mask

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: (C,H,W)
        return tensor * self.mask


def _default_shape(dataset_name: str):
    if dataset_name == "CIFAR10":
        return 3, 32, 32
    if dataset_name in {"MNIST", "FASHIONMNIST"}:
        return 1, 28, 28
    raise ValueError("Shape must be specified for custom datasets")


def _find_split_dir(root: str, candidates):
    """Return the first existing subdirectory from candidates under root."""
    for name in candidates:
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def _make_vertical_masks(
    dataset_name: str,
    num_clients: int,
    *,
    num_channels: Optional[int] = None,
    img_size: Optional[int] = None,
) -> list:
    """Create column-wise masks splitting the input width into num_clients slices."""
    if num_channels is not None and img_size is not None:
        c, h, w = num_channels, img_size, img_size
    else:
        c, h, w = _default_shape(dataset_name)
    masks = []
    # Split width equally; last client takes remainder
    base = w // num_clients
    remainder = w - base * num_clients
    start = 0
    for i in range(num_clients):
        width_i = base + (1 if i < remainder else 0)
        end = start + width_i
        mask = torch.zeros((c, h, w), dtype=torch.float32)
        mask[:, :, start:end] = 1.0
        masks.append(mask)
        start = end
    return masks


def _build_custom_transforms(num_channels: int, img_size: int):
    ops = [transforms.Resize((img_size, img_size))]
    if num_channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * num_channels, [0.5] * num_channels),
        ]
    )
    return transforms.Compose(ops)


class ClassBasedDataset(torch.utils.data.Dataset):
    """Filter dataset to only include samples from a specific class."""
    def __init__(self, base_dataset, target_class: int):
        self.base = base_dataset
        self.target_class = target_class
        # Filter indices for the target class
        self.indices = [i for i in range(len(base_dataset)) if base_dataset[i][1] == target_class]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


class ImbalancedClassDataset(torch.utils.data.Dataset):
    """Dataset with imbalanced class distribution per client.
    
    Each client gets:
    - 85% of their primary class
    - 5% of each other class
    """
    def __init__(self, base_dataset, client_id: int, num_clients: int, num_classes: int, seed: int = 42):
        self.base = base_dataset
        self.client_id = client_id
        self.num_clients = num_clients
        self.num_classes = num_classes
        
        # Assign primary class to client (client 0 -> class 0, client 1 -> class 1, etc.)
        primary_class = client_id % num_classes
        
        # Collect all samples grouped by class
        class_indices = {cls: [] for cls in range(num_classes)}
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            class_indices[int(label)].append(i)
        
        # Create imbalanced split: 85% primary, 5% each other class
        self.indices = []
        
        # Use a generator with seed for reproducibility
        import random
        rng = random.Random(seed + client_id)
        
        # Get 85% of primary class
        primary_class_indices = class_indices[primary_class].copy()
        rng.shuffle(primary_class_indices)
        primary_count = int(len(primary_class_indices) * 0.85)
        self.indices.extend(primary_class_indices[:primary_count])
        
        # Get 5% of each other class
        for cls in range(num_classes):
            if cls != primary_class:
                other_class_indices = class_indices[cls].copy()
                rng.shuffle(other_class_indices)
                other_count = max(1, int(len(other_class_indices) * 0.05))  # At least 1 sample
                self.indices.extend(other_class_indices[:other_count])
        
        # Shuffle all indices together
        rng.shuffle(self.indices)
        
        # Log distribution
        class_counts = {cls: 0 for cls in range(num_classes)}
        for idx in self.indices:
            _, label = base_dataset[idx]
            class_counts[int(label)] += 1
        
        total = len(self.indices)
        print(f"Client {client_id} (primary class {primary_class}): {total} total samples")
        for cls in range(num_classes):
            count = class_counts[cls]
            percentage = 100.0 * count / total if total > 0 else 0
            marker = " (PRIMARY)" if cls == primary_class else ""
            print(f"  Class {cls}: {count} samples ({percentage:.1f}%){marker}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def load_data(
    dataset_name="MNIST",
    num_clients=3,
    batch_size=32,
    partition_type: str = "horizontal",
    dataset_path: Optional[str] = None,
    img_size: Optional[int] = None,
    num_channels: Optional[int] = None,
):
    """
    Load and split dataset for federated learning.
    
    Args:
        dataset_name: Built-in dataset name or "CUSTOM" for ImageFolder inputs.
        num_clients: Number of client partitions.
        batch_size: Batch size for data loaders.
        partition_type: "horizontal" (default), "vertical" (feature-based), or "class_vertical" (class-based).
        dataset_path: Root folder for custom datasets (expects train/ and val/).
        img_size: Target square resolution (required for custom datasets).
        num_channels: Number of channels (required for custom datasets).
        
    Returns:
        List of DataLoaders (train), one for each client, and a single test DataLoader.
    """
    # Define transforms
    dataset_key = dataset_name.upper()

    if dataset_key == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_key == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_key == "FASHIONMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_key == "CUSTOM":
        if dataset_path is None:
            raise ValueError("dataset_path is required when using CUSTOM dataset")
        if img_size is None or num_channels is None:
            raise ValueError("img_size and num_channels must be provided for CUSTOM dataset")
        transform = _build_custom_transforms(num_channels, img_size)
        train_dir = _find_split_dir(dataset_path, ["train", "Train", "training", "Training"])
        val_dir = _find_split_dir(dataset_path, ["val", "Val", "valid", "Valid", "validation", "Validation"])
        test_dir = _find_split_dir(dataset_path, ["test", "Test"])
        if train_dir is None:
            raise FileNotFoundError(
                f"Custom dataset path must contain a train/ directory (looked in {dataset_path})"
            )
        eval_dir = test_dir or val_dir
        if eval_dir is None:
            raise FileNotFoundError(
                "Custom dataset path must contain either a val/ (or valid/) directory or a test/ directory"
            )
        dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        test_dataset = datasets.ImageFolder(root=eval_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if partition_type == "vertical":
        if dataset_key == "CUSTOM" and (num_channels is None or img_size is None):
            raise ValueError("Vertical partitioning for CUSTOM datasets requires --img-size and --num-channels")
        # Each client gets full dataset but with a different feature mask applied post-normalization
        masks = _make_vertical_masks(
            dataset_key,
            num_clients,
            num_channels=num_channels,
            img_size=img_size,
        )
        client_datasets = [
            MaskedDataset(dataset, post_transform=FeatureMaskTransform(mask))
            for mask in masks
        ]
        client_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in client_datasets
        ]
    elif partition_type == "class_vertical":
        # Each client gets imbalanced distribution: 85% of primary class, 5% of each other class
        # First, determine available classes
        if hasattr(dataset, 'classes'):
            # ImageFolder dataset
            num_classes = len(dataset.classes)
            class_names = dataset.classes
        else:
            # For other datasets, infer from labels
            all_labels = [dataset[i][1] for i in range(len(dataset))]
            unique_labels = sorted(set(all_labels))
            num_classes = len(unique_labels)
            class_names = [f"Class {i}" for i in unique_labels]
        
        if num_clients > num_classes:
            raise ValueError(f"Number of clients ({num_clients}) cannot exceed number of classes ({num_classes}) for class_vertical partitioning")
        
        # Create imbalanced datasets: 85% primary class, 5% each other class
        client_datasets = []
        for client_id in range(num_clients):
            client_dataset = ImbalancedClassDataset(
                dataset, 
                client_id=client_id,
                num_clients=num_clients,
                num_classes=num_classes,
                seed=42  # Fixed seed for reproducibility
            )
            client_datasets.append(client_dataset)
        
        client_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in client_datasets
        ]
    else:
        # Horizontal: equal random split of samples
        total_size = len(dataset)
        client_sizes = [total_size // num_clients] * num_clients
        client_sizes[-1] += total_size - sum(client_sizes)  # Handle remainder
        client_datasets = random_split(dataset, client_sizes)
        client_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in client_datasets
        ]
    
    # Test dataset
    if dataset_key == "MNIST":
        test_dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )
    elif dataset_key == "CIFAR10":
        test_dataset = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )
    elif dataset_key == "FASHIONMNIST":
        test_dataset = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )
    elif dataset_key == "CUSTOM":
        # Already loaded above
        pass

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader


def train_epoch(net, trainloader, device, epochs=1, debug=False, label_smoothing=0.1):
    """Train the model for one or more epochs.
    
    Args:
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = recommended for single-class clients)
                         This prevents overconfidence when clients only see one class.
    """
    # Use label smoothing to prevent overconfidence when clients only see one class
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Use different learning rates for feature extractor vs classifier
    # Lower LR for final layer to prevent it from becoming too confident
    feature_params = []
    classifier_params = []
    for name, param in net.named_parameters():
        # Identify final classification layer (varies by architecture)
        is_classifier = (
            'fc2' in name or 'fc2_hidden' in name or  # SimpleNet (both layers)
            (name.endswith('.fc.weight') or name.endswith('.fc.bias')) or  # ResNet (old format)
            ('.fc.' in name and ('weight' in name or 'bias' in name)) or  # ResNet (Sequential format - any layer)
            (name.endswith('.classifier.weight') or name.endswith('.classifier.bias'))  # Other architectures
        )
        if is_classifier:
            classifier_params.append(param)
        else:
            feature_params.append(param)
    
    # Create optimizer with differential learning rates if we have classifier params
    # Use lower base LR to prevent mode collapse
    base_lr = 0.001 if label_smoothing > 0 else 0.01  # Lower LR for class_vertical
    
    if classifier_params and label_smoothing > 0:
        optimizer = torch.optim.SGD([
            {'params': feature_params, 'lr': base_lr, 'momentum': 0.9, 'weight_decay': 1e-4},
            {'params': classifier_params, 'lr': base_lr * 0.1, 'momentum': 0.9, 'weight_decay': 1e-4}  # 10x lower LR for classifier
        ])
        if debug:
            print(f"  Using differential LR: features={base_lr}, classifier={base_lr * 0.1}")
    else:
        # Fallback to single LR if no classifier params found or no label smoothing
        optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
        if debug:
            print(f"  Using uniform LR: {base_lr}")
    
    # Gradient clipping to prevent explosion
    max_grad_norm = 1.0
    
    net.train()
    total_batches = len(trainloader)
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass - get raw logits (no softmax)
            outputs = net(images)
            
            # Debug: Log first batch statistics
            if debug and batch_idx == 0:
                print(f"\n  {'='*50}")
                print(f"  First Batch Analysis:")
                print(f"  {'='*50}")
                print(f"  Logits shape: {outputs.shape}, dtype: {outputs.dtype}")
                print(f"  Labels: {labels.cpu().numpy()}")
                print(f"  Logits stats: mean={outputs.mean().item():.6f}, std={outputs.std().item():.6f}")
                print(f"  Logits range: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
                print(f"  Logits per class (mean): {outputs.mean(dim=0).cpu().numpy()}")
                
                # Verify no softmax applied
                row_sums = outputs.sum(dim=1)
                if torch.allclose(row_sums, torch.ones_like(row_sums) * outputs.shape[1], atol=0.1):
                    print(f"  ✓ Outputs are raw logits (not probabilities)")
                else:
                    print(f"  ⚠️  WARNING: Outputs might be in probability space!")
                
                # Check for class distribution in batch
                unique_labels, counts = torch.unique(labels, return_counts=True)
                print(f"  Classes present: {dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))}")
                
                # Check if batch is single-class
                if len(unique_labels) == 1:
                    print(f"  ⚠️  WARNING: Batch contains only ONE class ({unique_labels[0].item()})!")
                print(f"  {'='*50}\n")
            
            # CrossEntropyLoss applies softmax internally - this is correct
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            
            # Debug: Log gradients after first batch
            if debug and batch_idx == 0:
                from comprehensive_diagnostics import log_gradient_norms_detailed
                # Note: client_id and round_num not available here, use placeholder
                grad_norms = log_gradient_norms_detailed(net, round_num=0, client_id=-1)
                
                # Verify optimizer will actually update
                has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
                if not has_grads:
                    print(f"  ⚠️  CRITICAL: No valid gradients found! Optimizer step() will not update parameters!")
            
            optimizer.step()
            running_loss += loss.item()
            
            # Log progress every 10% of batches
            if (batch_idx + 1) % max(1, total_batches // 10) == 0 or (batch_idx + 1) == total_batches:
                progress = 100.0 * (batch_idx + 1) / total_batches
                avg_loss = running_loss / (batch_idx + 1)
                print(f"  Training progress: {progress:.1f}% ({batch_idx + 1}/{total_batches} batches, avg loss: {avg_loss:.4f})")
        
        print(f"  Epoch {epoch + 1}/{epochs} complete, average loss: {running_loss / total_batches:.4f}")


def test(net, testloader, device):
    """Evaluate the model on test data."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    
    total_batches = len(testloader)
    print(f"  Evaluating on {total_batches} test batches...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Log progress every 25% of batches
            if (batch_idx + 1) % max(1, total_batches // 4) == 0 or (batch_idx + 1) == total_batches:
                progress = 100.0 * (batch_idx + 1) / total_batches
                print(f"  Evaluation progress: {progress:.1f}% ({batch_idx + 1}/{total_batches} batches)")
    
    accuracy = correct / total
    loss = loss / len(testloader)
    return loss, accuracy

