"""
DB <-> tensor bridge for MIA evaluation.

This is the only non-CLI module in the MIA stack that knows about the
contributions DB, the canonical torchvision datasets, and disk layout. It is
deliberately kept thin so that ``mia.py`` (pure attacks) and ``mia_shadow.py``
(shadow training) stay free of FL-specific assumptions.

The transform construction below MUST stay in sync with ``utils.load_data``
(utils.py:211-243). If those normalizations diverge, MIA AUC will silently
inflate from a transform mismatch rather than from genuine membership leakage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Set

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sample_ids import WithSampleIds

if TYPE_CHECKING:
    from contributions_db import ContributionDB


_SUPPORTED = ("MNIST", "CIFAR10", "FASHIONMNIST")


def _build_transforms(dataset_name: str) -> transforms.Compose:
    """Canonical transform for a built-in dataset; mirrors utils.py:211-243."""
    key = dataset_name.upper()
    if key == "MNIST":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    if key == "CIFAR10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    if key == "FASHIONMNIST":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    raise ValueError(
        f"Unsupported dataset for MIA: {dataset_name}. Supported: {_SUPPORTED}"
    )


def _load_canonical_dataset(dataset_name: str, *, train_split: bool):
    """Load the unmodified torchvision dataset (no FL partitioning)."""
    key = dataset_name.upper()
    transform = _build_transforms(key)
    if key == "MNIST":
        return datasets.MNIST(
            root="./data", train=train_split, download=True, transform=transform
        )
    if key == "CIFAR10":
        return datasets.CIFAR10(
            root="./data", train=train_split, download=True, transform=transform
        )
    if key == "FASHIONMNIST":
        return datasets.FashionMNIST(
            root="./data", train=train_split, download=True, transform=transform
        )
    raise ValueError(
        f"Unsupported dataset for MIA: {dataset_name}. Supported: {_SUPPORTED}"
    )


def get_train_pool_size(dataset_name: str) -> int:
    return len(_load_canonical_dataset(dataset_name, train_split=True))


def get_test_pool_size(dataset_name: str) -> int:
    return len(_load_canonical_dataset(dataset_name, train_split=False))


def collect_member_sample_ids(db: "ContributionDB") -> Set[int]:
    """
    Union of training sample IDs across every client contribution in the DB.

    Reads ``sample_ids`` directly from each ``*_sample_manifest.json`` so we
    don't need to parse training_trace.
    """
    members: Set[int] = set()
    contrib_dir = Path(db.contributions_dir)
    for round_dir in sorted(contrib_dir.glob("round_*")):
        if not round_dir.is_dir():
            continue
        for client_dir in sorted(round_dir.glob("client_*")):
            if not client_dir.is_dir():
                continue
            for manifest_path in client_dir.glob("*_sample_manifest.json"):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                for sid in manifest.get("sample_ids", []) or []:
                    members.add(int(sid))
    return members


def collect_forgotten_sample_ids(db: "ContributionDB") -> Set[int]:
    """All sample IDs currently marked as withdrawn (sum across contributions)."""
    forgotten: Set[int] = set()
    active = (db.withdrawals or {}).get("sample_withdrawals", {}).get("active", {})
    for entry in active.values():
        for sid in entry.get("sample_ids", []) or []:
            forgotten.add(int(sid))
    return forgotten


def build_evaluation_loader(
    dataset_name: str,
    sample_ids: Iterable[int],
    *,
    train_split: bool = True,
    batch_size: int = 256,
    num_workers: int = 0,
) -> DataLoader:
    """
    DataLoader yielding (image, label, sample_id) for a fixed set of indices.

    The yielded sample_id is the raw position in the chosen split. Members and
    non-members are forward-passed through separate calls, so callers do not
    need to namespace IDs across splits.
    """
    ids: List[int] = [int(s) for s in sample_ids]
    if not ids:
        raise ValueError("sample_ids must be non-empty")
    base = _load_canonical_dataset(dataset_name, train_split=train_split)
    pool = len(base)
    if min(ids) < 0 or max(ids) >= pool:
        raise ValueError(
            f"sample_id out of range [0, {pool}) for {dataset_name} "
            f"({'train' if train_split else 'test'})"
        )
    subset = Subset(base, ids)
    wrapped = WithSampleIds(subset, list(ids))
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def sample_test_set_non_members(
    dataset_name: str, n: int, *, seed: int = 42
) -> List[int]:
    """Deterministically sample ``n`` distinct test-split positions."""
    pool = get_test_pool_size(dataset_name)
    if n > pool:
        raise ValueError(f"Requested {n} non-members but test set has {pool}")
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(pool, size=n, replace=False))


def sample_train_pool_non_members(
    dataset_name: str,
    exclude_ids: Iterable[int],
    n: int,
    *,
    seed: int = 42,
) -> List[int]:
    """
    Deterministically sample ``n`` distinct training-split positions excluding
    ``exclude_ids``. Use this for LiRA non-members so they overlap with the
    shadow models' training pool (test split has no shadow signals).
    """
    pool = get_train_pool_size(dataset_name)
    excluded_set = {int(x) for x in exclude_ids}
    excluded = np.fromiter(excluded_set, dtype=np.int64, count=len(excluded_set))
    candidates = np.setdiff1d(np.arange(pool, dtype=np.int64), excluded)
    if n > candidates.size:
        raise ValueError(
            f"Requested {n} non-members but only {candidates.size} candidates "
            f"({pool} pool minus {excluded.size} excluded)"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=n, replace=False)
    return sorted(int(x) for x in chosen)
