"""
Stable sample identity for federated training (sample-level unlearning).

Each training example in the canonical *training* split is assigned an integer
``global_sample_id`` in ``[0, N)`` where ``N`` is ``len(train_dataset)`` and
ordering matches PyTorch's dataset indexing (torchvision/ImageFolder order).

Client and server agree by sharing:
  - ``dataset`` name (e.g. MNIST, CIFAR10),
  - ``partition_type`` and partition parameters (e.g. ``partition_seed`` for horizontal splits),
  - ``sample_id_scheme_version`` (increment when semantics change).

Horizontal splits use a fixed RNG seed so the same client always receives the
same subset of global ids across runs.

Per-sample gradients: full tensors are not stored in metrics. Use
``influence_replay.replay_local_epoch_and_collect_per_sample_gradients`` with saved
``training_trace`` and incoming global checkpoints (see ``ContributionDB``).
"""

from __future__ import annotations

import torch

SAMPLE_ID_SCHEME_VERSION = 1

# Structure of ``training_trace`` in client fit metrics (instrumented training loop).
# Keys: ``batches`` — ordered list of [sample_id, ...] per mini-batch (DataLoader iteration order);
# ``per_sample_loss`` — flat list of {sample_id, loss} in that same order;
# ``per_sample_grad_norm`` — optional flat list of {sample_id, grad_norm} when mode is ``full``.
TRAINING_TRACE_VERSION = 1

# Default seed for torch.utils.data.random_split (horizontal partitioning).
DEFAULT_PARTITION_SEED = 42

# ImbalancedClassDataset (class_vertical) uses this seed internally.
IMBALANCED_CLASS_PARTITION_SEED = 42


def format_global_sample_id(dataset_name: str, global_index: int) -> str:
    """Human-readable id: ``DATASET|train|00001234`` (8-digit index)."""
    return f"{dataset_name.upper()}|train|{global_index:08d}"


class WithSampleIds(torch.utils.data.Dataset):
    """Wraps a dataset so each item is ``(x, y, global_sample_id)``."""

    def __init__(self, inner: torch.utils.data.Dataset, global_ids: list[int]):
        if len(inner) != len(global_ids):
            raise ValueError(
                f"WithSampleIds: len(dataset)={len(inner)} != len(global_ids)={len(global_ids)}"
            )
        self.inner = inner
        self.global_ids = global_ids

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        x, y = self.inner[idx]
        return x, y, int(self.global_ids[idx])


def get_global_index_map(dataset) -> list[int]:
    """
    For each local index ``i`` in ``dataset``, return the canonical global
    training index that sample corresponds to.

    Works with Subset (horizontal splits), ImbalancedClassDataset / ClassBasedDataset,
    MaskedDataset (vertical FL — full dataset, identity indices), and raw datasets
    (identity mapping).
    """
    from torch.utils.data import Subset

    from utils import ClassBasedDataset, ImbalancedClassDataset, MaskedDataset

    if isinstance(dataset, Subset):
        return list(dataset.indices)
    if isinstance(dataset, ImbalancedClassDataset):
        return list(dataset.indices)
    if isinstance(dataset, ClassBasedDataset):
        return list(dataset.indices)
    if isinstance(dataset, MaskedDataset):
        return list(range(len(dataset.base)))
    return list(range(len(dataset)))
