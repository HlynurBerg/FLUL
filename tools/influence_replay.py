"""
Per-sample gradients via **replay** (recommended for this codebase).

Full per-sample gradients are not stored in federated metrics: they are too large for
ResNet-scale models. Strategies:

1. **Per-sample loss only** — recorded in ``training_trace`` when the client runs with
   ``--training-trace loss`` (cheap; enough for many MIA-style analyses).
2. **Projections / sketches** — not implemented here (middle ground).
3. **Replay (this module)** — on demand, load the **incoming global** checkpoint for
   that round, rebuild the client's ``DataLoader`` with the same ``load_data`` settings,
   and re-run training using the **exact** mini-batch order stored in
   ``training_trace["batches"]``. Then compute per-sample gradient norms (or extend to
   full per-parameter grads) inside ``train_epoch`` without ever serializing them during
   the live FL round.

Requirements:

- Server writes ``initial_global_params.pkl`` (round 0 incoming) — see
  ``ContributionDB.save_initial_global_parameters``.
- Aggregated models after each round live under ``round_XXXX/*_aggregated_*_params.pkl``.
- Per-sample fields may live in ``*_sample_manifest.json``; ``get_contribution_info``
  merges them into ``metrics`` automatically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import default_collate

from model import set_parameters
from sample_ids import WithSampleIds
from utils import train_epoch


class ReplayDataLoader:
    """
    Yields the same mini-batch grouping and order as stored in ``training_trace``,
    without relying on DataLoader shuffle. Each batch is collated like a normal
    ``DataLoader`` (including ``(images, labels, sample_id)`` tensors).
    """

    def __init__(self, with_sample_ids_dataset: WithSampleIds, batches_spec: List[List[int]]):
        if not isinstance(with_sample_ids_dataset, WithSampleIds):
            raise TypeError("ReplayDataLoader expects a WithSampleIds dataset (from load_data).")
        self.dataset = with_sample_ids_dataset
        self.batches_spec = batches_spec
        self.gid_to_local = {g: i for i, g in enumerate(with_sample_ids_dataset.global_ids)}

    def __len__(self) -> int:
        return len(self.batches_spec)

    def __iter__(self):
        for gids in self.batches_spec:
            rows = []
            for g in gids:
                if g not in self.gid_to_local:
                    raise KeyError(
                        f"Global sample id {g} is not in this client's partition; "
                        "cannot replay (check partition_seed / partition_type)."
                    )
                rows.append(self.dataset[self.gid_to_local[g]])
            yield default_collate(rows)


def replay_local_epoch_and_collect_per_sample_gradients(
    net: torch.nn.Module,
    trainloader,
    device: torch.device,
    *,
    contribution_db,
    round_num: int,
    client_id: int,
    label_smoothing: float = 0.0,
    per_sample_grad_norms: bool = True,
) -> Dict[str, Any]:
    """
    Restore incoming global weights for ``round_num``, replay one local epoch in the
    stored batch order, and optionally fill per-sample loss and per-sample gradient norms.

    ``trainloader`` must be built with the **same** ``load_data`` settings as the original run.

    Returns a **new** ``training_trace``-style dict (batches, per_sample_loss, and
    optionally per_sample_grad_norm). Does not mutate contribution files.
    """
    meta = contribution_db.get_contribution_info(round_num, client_id)
    if meta is None:
        raise FileNotFoundError(
            f"No contribution metadata for round {round_num}, client {client_id}"
        )
    metrics = meta.get("metrics") or {}
    trace = metrics.get("training_trace") or {}
    batches = trace.get("batches")
    if not batches:
        raise ValueError(
            "Contribution metadata has no training_trace['batches']. "
            "Train clients with --training-trace loss (or full)."
        )

    incoming = contribution_db.load_incoming_global_for_round(round_num)
    if incoming is None:
        raise FileNotFoundError(
            f"Could not load incoming global model for round {round_num}. "
            "For round 0, ensure the server saved initial_global_params.pkl; "
            "for round > 0, ensure round_{R-1} aggregated checkpoint exists."
        )

    set_parameters(net, incoming)

    replay_loader = ReplayDataLoader(trainloader.dataset, batches)
    training_stats: Dict[str, Any] = {"batches": [], "per_sample_loss": []}
    if per_sample_grad_norms:
        training_stats["per_sample_grad_norm"] = []

    train_epoch(
        net,
        replay_loader,
        device,
        epochs=1,
        debug=False,
        label_smoothing=label_smoothing,
        sample_ids_accumulator=None,
        training_stats=training_stats,
        per_sample_grad_norms=per_sample_grad_norms,
    )

    return {
        "round": round_num,
        "client_id": client_id,
        "incoming_global_source": "initial" if round_num == 0 else f"aggregated_after_round_{round_num - 1}",
        "training_trace": training_stats,
    }


def get_training_trace_from_contribution(metadata: dict) -> Optional[dict]:
    """Return ``metrics['training_trace']`` if present."""
    return (metadata.get("metrics") or {}).get("training_trace")
