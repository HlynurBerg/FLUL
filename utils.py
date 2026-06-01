"""Utility functions for data loading and preprocessing."""
import os
from typing import Optional

import sample_ids
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
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


def _extract_targets(dataset) -> list:
    """Extract integer labels for every sample. Uses ``.targets`` when available
    (torchvision MNIST/CIFAR/FashionMNIST/ImageFolder), falling back to a full
    iteration as a last resort."""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return [int(dataset[i][1]) for i in range(len(dataset))]
    if isinstance(targets, torch.Tensor):
        return targets.tolist()
    return [int(t) for t in targets]


def _dirichlet_partition_indices(
    dataset, num_clients: int, alpha: float, seed: int
) -> list:
    """Per-class Dirichlet(α) label-skew partitioning.

    For each class c, draw proportions ``p_c ~ Dirichlet(α, ..., α)`` over
    ``num_clients`` and allocate that class's samples accordingly. Smaller α →
    more skew (each class concentrates on few clients). α → ∞ approaches IID.

    Allocation is disjoint (every sample goes to exactly one client), so total
    coverage matches a horizontal split. Determined by ``seed``.
    """
    if alpha <= 0:
        raise ValueError(f"dirichlet_alpha must be > 0, got {alpha}")
    rng = np.random.default_rng(seed)
    targets = _extract_targets(dataset)

    by_class: dict = {}
    for i, y in enumerate(targets):
        by_class.setdefault(int(y), []).append(i)

    client_indices = [[] for _ in range(num_clients)]
    for cls in sorted(by_class):
        cls_indices = np.array(by_class[cls], dtype=np.int64)
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
        splits = np.split(cls_indices, cuts)
        for k, split in enumerate(splits):
            client_indices[k].extend(int(x) for x in split.tolist())

    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    empty_clients = [k for k, ix in enumerate(client_indices) if len(ix) == 0]
    if empty_clients:
        print(
            f"[load_data] WARNING: Dirichlet(α={alpha}) left clients {empty_clients} "
            f"with zero samples. Consider raising α or reducing num_clients."
        )

    return client_indices


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

    Each client takes ``primary_share`` of its primary class's samples plus
    ``secondary_share`` of every other class's samples. Defaults reproduce the
    legacy 85% / 5% split. Setting ``secondary_share=0`` yields a pure-class
    client (no samples from other classes — used to probe class collapse).
    """
    def __init__(
        self,
        base_dataset,
        client_id: int,
        num_clients: int,
        num_classes: int,
        seed: Optional[int] = None,
        primary_share: float = 0.85,
        secondary_share: float = 0.05,
    ):
        if seed is None:
            seed = sample_ids.IMBALANCED_CLASS_PARTITION_SEED
        if not 0.0 <= primary_share <= 1.0:
            raise ValueError(f"primary_share must be in [0, 1], got {primary_share}")
        if not 0.0 <= secondary_share <= 1.0:
            raise ValueError(f"secondary_share must be in [0, 1], got {secondary_share}")
        self.base = base_dataset
        self.client_id = client_id
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.primary_share = primary_share
        self.secondary_share = secondary_share

        # Assign primary class to client (client 0 -> class 0, client 1 -> class 1, etc.)
        primary_class = client_id % num_classes

        # Collect all samples grouped by class
        class_indices = {cls: [] for cls in range(num_classes)}
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            class_indices[int(label)].append(i)

        # Create imbalanced split per primary_share / secondary_share
        self.indices = []

        # Use a generator with seed for reproducibility
        import random
        rng = random.Random(seed + client_id)

        # Take primary_share of primary class
        primary_class_indices = class_indices[primary_class].copy()
        rng.shuffle(primary_class_indices)
        primary_count = int(len(primary_class_indices) * primary_share)
        self.indices.extend(primary_class_indices[:primary_count])

        # Take secondary_share of each other class (0 means take none — pure-class client)
        for cls in range(num_classes):
            if cls == primary_class:
                continue
            if secondary_share <= 0.0:
                continue
            other_class_indices = class_indices[cls].copy()
            rng.shuffle(other_class_indices)
            other_count = max(1, int(len(other_class_indices) * secondary_share))
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
    partition_seed: int = None,
    dirichlet_alpha: Optional[float] = None,
    primary_share: float = 0.85,
    secondary_share: float = 0.05,
):
    """
    Load and split dataset for federated learning.

    Args:
        dataset_name: Built-in dataset name or "CUSTOM" for ImageFolder inputs.
        num_clients: Number of client partitions.
        batch_size: Batch size for data loaders.
        partition_type: "horizontal" (default, IID random split), "vertical"
            (feature-mask), "class_vertical" (per-client class imbalance,
            controlled by ``primary_share`` / ``secondary_share``), or
            "dirichlet" (per-class Dirichlet(α) label-skew, controlled by
            ``dirichlet_alpha``).
        dataset_path: Root folder for custom datasets (expects train/ and val/).
        img_size: Target square resolution (required for custom datasets).
        num_channels: Number of channels (required for custom datasets).
        partition_seed: RNG seed for horizontal random_split AND dirichlet
            sampling. Stable across runs to keep global sample ids reproducible.
        dirichlet_alpha: Concentration for ``partition_type="dirichlet"``.
            Smaller → more label skew (e.g. 0.1 ≈ extreme non-IID); larger →
            closer to IID (e.g. 1000 ≈ horizontal). Required when
            ``partition_type="dirichlet"``.
        primary_share: Per-client share of the client's primary class for
            ``class_vertical`` (default 0.85, the legacy value).
        secondary_share: Per-client share of every non-primary class for
            ``class_vertical`` (default 0.05). Set to 0 for pure-class clients.

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
    
    if partition_seed is None:
        partition_seed = sample_ids.DEFAULT_PARTITION_SEED
    
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
    elif partition_type == "class_vertical":
        # Each client gets imbalanced distribution per (primary_share, secondary_share).
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

        client_datasets = []
        for client_id in range(num_clients):
            client_dataset = ImbalancedClassDataset(
                dataset,
                client_id=client_id,
                num_clients=num_clients,
                num_classes=num_classes,
                seed=sample_ids.IMBALANCED_CLASS_PARTITION_SEED,
                primary_share=primary_share,
                secondary_share=secondary_share,
            )
            client_datasets.append(client_dataset)
    elif partition_type == "dirichlet":
        if dirichlet_alpha is None:
            raise ValueError(
                "partition_type='dirichlet' requires dirichlet_alpha (e.g. 0.1, 1.0, 10.0)."
            )
        indices_per_client = _dirichlet_partition_indices(
            dataset, num_clients, float(dirichlet_alpha), int(partition_seed)
        )
        client_datasets = [Subset(dataset, idx) for idx in indices_per_client]
    else:
        # Horizontal: equal random split of samples (seeded for stable global sample ids)
        total_size = len(dataset)
        client_sizes = [total_size // num_clients] * num_clients
        client_sizes[-1] += total_size - sum(client_sizes)  # Handle remainder
        g = torch.Generator()
        g.manual_seed(int(partition_seed))
        client_datasets = random_split(dataset, client_sizes, generator=g)
    
    client_datasets = [
        sample_ids.WithSampleIds(ds, sample_ids.get_global_index_map(ds))
        for ds in client_datasets
    ]
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


def _total_grad_norm(parameters):
    """L2 norm of gradients over all parameters (per-sample gradient norm)."""
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().data.pow(2).sum().item())
    return total ** 0.5


def train_epoch(
    net,
    trainloader,
    device,
    epochs=1,
    debug=False,
    label_smoothing=0.1,
    sample_ids_accumulator=None,
    training_stats=None,
    per_sample_grad_norms=False,
):
    """Train the model for one or more epochs.
    
    Args:
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = recommended for single-class clients)
                         This prevents overconfidence when clients only see one class.
        sample_ids_accumulator: If provided, extend with each batch's global sample ids in **iteration order**
            (includes duplicates across batches if any; use ``training_stats`` for batch structure).
        training_stats: If provided, must be a dict; it is filled with:
            ``batches`` (ordered list of per-batch sample-id lists, reflecting DataLoader shuffle),
            ``per_sample_loss`` (flat list of ``{sample_id, loss}`` in iteration order),
            and optionally ``per_sample_grad_norm`` (same shape when ``per_sample_grad_norms`` is True).
        per_sample_grad_norms: If True, run one backward per example in each batch to record grad norms,
            then a second forward/backward for the optimizer step (expensive).
    """
    
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
    
    if per_sample_grad_norms and training_stats is None:
        training_stats = {"batches": [], "per_sample_loss": [], "per_sample_grad_norm": []}
    if training_stats is not None:
        training_stats.setdefault("batches", [])
        training_stats.setdefault("per_sample_loss", [])
        if per_sample_grad_norms:
            training_stats.setdefault("per_sample_grad_norm", [])
    
    net.train()
    total_batches = len(trainloader)
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(trainloader):
            if len(batch) == 3:
                images, labels, sample_id_batch = batch
                images, labels = images.to(device), labels.to(device)
                sids = sample_id_batch.detach().cpu().tolist()
                if sample_ids_accumulator is not None:
                    sample_ids_accumulator.extend(sids)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                sids = None
            
            if training_stats is not None and sids is None:
                raise ValueError(
                    "training_stats / per-sample metrics require DataLoader batches with sample ids "
                    "(image, label, global_sample_id); enable WithSampleIds in load_data."
                )
            
            optimizer.zero_grad()
            
            # Forward pass - get raw logits (no softmax)
            outputs = net(images)
            
            # Per-example loss (same math as mean CE used for the optimizer step)
            loss_vec = F.cross_entropy(
                outputs, labels, label_smoothing=label_smoothing, reduction="none"
            )
            
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
            
            if training_stats is not None:
                training_stats["batches"].append([int(x) for x in sids])
                lv = loss_vec.detach().cpu()
                for j, sid in enumerate(sids):
                    training_stats["per_sample_loss"].append(
                        {"sample_id": int(sid), "loss": float(lv[j].item())}
                    )
            
            if per_sample_grad_norms:
                norms = []
                bsz = images.size(0)
                for j in range(bsz):
                    optimizer.zero_grad()
                    loss_vec[j].backward(retain_graph=(j < bsz - 1))
                    norms.append(_total_grad_norm(net.parameters()))
                if training_stats is not None:
                    for j, sid in enumerate(sids):
                        training_stats["per_sample_grad_norm"].append(
                            {"sample_id": int(sid), "grad_norm": float(norms[j])}
                        )
                optimizer.zero_grad()
                # Second forward at the same weights for the actual SGD step (graph was freed)
                outputs = net(images)
                loss_vec = F.cross_entropy(
                    outputs, labels, label_smoothing=label_smoothing, reduction="none"
                )
                loss = loss_vec.mean()
            else:
                loss = loss_vec.mean()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            
            # Debug: Log gradients after first batch
            if debug and batch_idx == 0:
                try:
                    from Information.comprehensive_diagnostics import log_gradient_norms_detailed
                except ImportError:
                    from Information.comprehensive_diagnostics import log_gradient_norms_detailed
                log_gradient_norms_detailed(net, round_num=0, client_id=-1)
                
                # Verify optimizer will actually update
                has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
                if not has_grads:
                    print(f"  ⚠️  CRITICAL: No valid gradients found! Optimizer step() will not update parameters!")
            
            optimizer.step()
            running_loss += float(loss.detach().item())
            
            # Log progress every 10% of batches
            if (batch_idx + 1) % max(1, total_batches // 10) == 0 or (batch_idx + 1) == total_batches:
                progress = 100.0 * (batch_idx + 1) / total_batches
                avg_loss = running_loss / (batch_idx + 1)
                print(f"  Training progress: {progress:.1f}% ({batch_idx + 1}/{total_batches} batches, avg loss: {avg_loss:.4f})")
        
        print(f"  Epoch {epoch + 1}/{epochs} complete, average loss: {running_loss / total_batches:.4f}")


def test(net, testloader, device):
    """Evaluate the model on test data.

    Accepts loaders that yield either ``(image, label)`` or
    ``(image, label, global_sample_id)`` batches — sample IDs are ignored when
    present, so this works with both vanilla torchvision loaders and the
    ``WithSampleIds``-wrapped loaders used by the unlearning + MIA pipelines.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    total_batches = len(testloader)
    print(f"  Evaluating on {total_batches} test batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
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

