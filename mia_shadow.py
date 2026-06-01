"""
Shadow-model training and signal-matrix building for offline LiRA.

Shadow models train centralized SGD on random ~50% subsets of the canonical
training split — never on the test split, which stays held out for non-member
evaluation. After each shadow trains, the scaled logit log(p_y) - log(1 - p_y)
is recorded for every training-split sample, so for each (sample, shadow)
pair we know whether the sample was IN or OUT and what signal the shadow
emits on it. compute_shadow_signal_matrix slices these into the per-sample
out-signal lists that mia.lira_offline_scores consumes.

Cache layout:
    cache_dir/<spec_hash>/
        spec.json
        shadow_NNN.npz   - per shadow:
            p0, p1, ...      param arrays in state_dict order
            n_params         scalar
            member_mask      bool, shape (pool_size,)
            scaled_logits    float32, shape (pool_size,)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mia import forward_with_ids
from mia_eval_sets import _load_canonical_dataset
from model import create_model, get_parameters
from sample_ids import SAMPLE_ID_SCHEME_VERSION, WithSampleIds
from utils import train_epoch


@dataclass
class ShadowSpec:
    dataset: str
    model_name: str = "simplenet"
    n_shadow: int = 10
    sampling_rate: float = 0.5
    epochs: int = 5
    label_smoothing: float = 0.0
    batch_size: int = 64
    seed: int = 42
    sample_id_scheme_version: int = SAMPLE_ID_SCHEME_VERSION

    def hash(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class ShadowResult:
    params: List[np.ndarray]
    member_mask: np.ndarray  # bool, shape (pool_size,)
    scaled_logits_full: np.ndarray  # float32, shape (pool_size,)

    @property
    def member_ids(self) -> List[int]:
        return np.where(self.member_mask)[0].tolist()


def _spec_dir(cache_dir, spec: ShadowSpec) -> Path:
    return Path(cache_dir) / spec.hash()


def _save_shadow(path: Path, result: ShadowResult) -> None:
    arrays: Dict[str, np.ndarray] = {
        f"p{i}": np.asarray(p) for i, p in enumerate(result.params)
    }
    arrays["n_params"] = np.asarray(len(result.params))
    arrays["member_mask"] = result.member_mask.astype(bool)
    arrays["scaled_logits"] = result.scaled_logits_full.astype(np.float32)
    np.savez_compressed(path, **arrays)


def _load_shadow(path: Path) -> ShadowResult:
    data = np.load(path, allow_pickle=False)
    n = int(data["n_params"])
    params = [np.asarray(data[f"p{i}"]) for i in range(n)]
    return ShadowResult(
        params=params,
        member_mask=data["member_mask"].astype(bool),
        scaled_logits_full=data["scaled_logits"].astype(np.float32),
    )


def _compute_full_scaled_logits(
    net,
    dataset_name: str,
    device,
    batch_size: int = 256,
) -> np.ndarray:
    """Score every training-split sample; return logits indexed by sample id."""
    base = _load_canonical_dataset(dataset_name, train_split=True)
    n = len(base)
    wrapped = WithSampleIds(base, list(range(n)))
    loader = DataLoader(wrapped, batch_size=batch_size, shuffle=False, num_workers=0)
    data = forward_with_ids(net, loader, device)
    out = np.zeros(n, dtype=np.float32)
    for sid, fields in data.items():
        out[sid] = fields["scaled_logit"]
    return out


def _generate_in_sets(spec: ShadowSpec, pool_size: int) -> List[List[int]]:
    """Deterministic per-shadow in-sets, derived from spec.seed."""
    rng = np.random.default_rng(spec.seed)
    n_in = int(round(spec.sampling_rate * pool_size))
    return [
        sorted(int(x) for x in rng.choice(pool_size, size=n_in, replace=False))
        for _ in range(spec.n_shadow)
    ]


def _train_one_shadow(
    spec: ShadowSpec,
    in_set_ids: List[int],
    pool_size: int,
    device,
) -> ShadowResult:
    base = _load_canonical_dataset(spec.dataset, train_split=True)
    subset = Subset(base, in_set_ids)
    loader = DataLoader(
        subset, batch_size=spec.batch_size, shuffle=True, num_workers=0
    )
    net = create_model(spec.model_name, spec.dataset).to(device)
    train_epoch(
        net,
        loader,
        device,
        epochs=spec.epochs,
        label_smoothing=spec.label_smoothing,
    )
    scaled_logits = _compute_full_scaled_logits(
        net, spec.dataset, device, batch_size=max(spec.batch_size, 256)
    )
    mask = np.zeros(pool_size, dtype=bool)
    mask[in_set_ids] = True
    return ShadowResult(
        params=get_parameters(net),
        member_mask=mask,
        scaled_logits_full=scaled_logits,
    )


def train_shadow_models(
    spec: ShadowSpec,
    cache_dir,
    device,
    *,
    force: bool = False,
) -> List[ShadowResult]:
    """
    Train (or load from cache) all shadows. Returns one ShadowResult per shadow.

    Caching is per-shadow: existing shadow_NNN.npz files are reused unless
    ``force`` is True.
    """
    spec_dir = _spec_dir(cache_dir, spec)
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_file = spec_dir / "spec.json"
    spec_dict = asdict(spec)
    if spec_file.exists():
        on_disk = json.loads(spec_file.read_text())
        if on_disk != spec_dict:
            raise RuntimeError(
                f"Spec hash collision in {spec_dir}: cached spec differs from "
                f"requested spec.\n  cached: {on_disk}\n  requested: {spec_dict}"
            )
    else:
        spec_file.write_text(json.dumps(spec_dict, indent=2))

    base = _load_canonical_dataset(spec.dataset, train_split=True)
    pool_size = len(base)
    in_sets = _generate_in_sets(spec, pool_size)

    results: List[ShadowResult] = []
    for i, in_set in enumerate(in_sets):
        path = spec_dir / f"shadow_{i:03d}.npz"
        if path.exists() and not force:
            results.append(_load_shadow(path))
            print(f"[mia_shadow] Shadow {i+1}/{spec.n_shadow}: cache hit ({path.name})")
            continue
        t0 = time.time()
        result = _train_one_shadow(spec, in_set, pool_size, device)
        _save_shadow(path, result)
        print(
            f"[mia_shadow] Shadow {i+1}/{spec.n_shadow}: trained in "
            f"{time.time() - t0:.1f}s  ({len(in_set)} in-set samples) -> {path.name}"
        )
        results.append(result)
    return results


def load_or_train_shadows(
    spec: ShadowSpec, cache_dir, device, *, force: bool = False
) -> List[ShadowResult]:
    """Alias — train_shadow_models already supports cache hits."""
    return train_shadow_models(spec, cache_dir, device, force=force)


def compute_shadow_signal_matrix(
    shadow_results: List[ShadowResult],
    eval_sample_ids: Iterable[int],
) -> Dict[int, List[float]]:
    """
    Per-sample list of scaled logits from shadows where the sample was OUT.

    Used as ``shadow_out_signals`` for ``mia.lira_offline_scores``.
    """
    out: Dict[int, List[float]] = {}
    for sid in eval_sample_ids:
        sid = int(sid)
        signals: List[float] = []
        for sr in shadow_results:
            if not sr.member_mask[sid]:
                signals.append(float(sr.scaled_logits_full[sid]))
        out[sid] = signals
    return out
