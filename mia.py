"""
Membership Inference Attack (MIA) library.

Pure attack functions: tensors in, numpy out. No DB or filesystem access.
Three attacks supported:
  1. Loss-threshold (Yeom et al. 2018)
  2. Modified-entropy (Song & Mittal 2021)
  3. LiRA offline (Carlini et al. 2022)

Convention: every attack returns scores where higher = more likely member.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

_EPS = 1e-12


@dataclass
class ThresholdResult:
    auc: float
    balanced_accuracy: float
    tpr_at_1pct_fpr: float
    tpr_at_0_1pct_fpr: float
    n_members: int
    n_nonmembers: int


def forward_with_ids(model, loader, device) -> Dict[int, Dict[str, float]]:
    """
    Single forward pass over a DataLoader yielding (image, label, sample_id).
    Returns dict {sid: {loss, scaled_logit, mod_entropy, label, prob_correct}}.

    model.eval() is called unconditionally — SimpleNet uses BatchNorm + Dropout
    and train-mode forwards would randomize per-sample losses.
    """
    model.eval()
    out: Dict[int, Dict[str, float]] = {}
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                raise ValueError(
                    "forward_with_ids requires (image, label, sample_id) batches. "
                    "Wrap the dataset with sample_ids.WithSampleIds."
                )
            images, labels, sids = batch
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            log_p = F.log_softmax(logits, dim=-1)
            p = log_p.exp()
            n = p.size(0)
            row = torch.arange(n, device=device)

            log_p_y = log_p[row, labels]
            p_y = p[row, labels].clamp(min=_EPS, max=1.0 - _EPS)

            loss = -log_p_y
            scaled_logit = log_p_y - torch.log1p(-p_y)

            # Modified entropy (Song & Mittal 2021):
            #   M = -(1 - p_y) log p_y  -  sum_{i != y} p_i log(1 - p_i)
            log1m_p = torch.log1p(-p.clamp(max=1.0 - _EPS))
            contrib = p * log1m_p
            term1 = -(1.0 - p_y) * log_p_y
            term2 = -(contrib.sum(dim=-1) - contrib[row, labels])
            mod_entropy = term1 + term2

            sids_list = (
                sids.tolist() if torch.is_tensor(sids) else [int(s) for s in sids]
            )
            loss_l = loss.cpu().tolist()
            sl_l = scaled_logit.cpu().tolist()
            me_l = mod_entropy.cpu().tolist()
            lab_l = labels.cpu().tolist()
            py_l = p_y.cpu().tolist()
            for i, sid in enumerate(sids_list):
                out[int(sid)] = {
                    "loss": loss_l[i],
                    "scaled_logit": sl_l[i],
                    "mod_entropy": me_l[i],
                    "label": int(lab_l[i]),
                    "prob_correct": py_l[i],
                }
    return out


def loss_attack_scores(
    per_sample_data: Dict[int, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Yeom et al.: score = -loss (lower loss => more likely member)."""
    ids = np.array(sorted(per_sample_data), dtype=np.int64)
    scores = np.array(
        [-per_sample_data[int(s)]["loss"] for s in ids], dtype=np.float64
    )
    return ids, scores


def modified_entropy_attack_scores(
    per_sample_data: Dict[int, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Song & Mittal: score = -mod_entropy (lower entropy => more likely member)."""
    ids = np.array(sorted(per_sample_data), dtype=np.int64)
    scores = np.array(
        [-per_sample_data[int(s)]["mod_entropy"] for s in ids], dtype=np.float64
    )
    return ids, scores


def lira_offline_scores(
    target_scaled_logits: Dict[int, float],
    shadow_out_signals: Dict[int, List[float]],
    *,
    fixed_variance: bool = True,
    min_out: int = 2,
    sigma_floor: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Offline LiRA score: per-sample one-sided z-score against the OUT distribution.

    Args:
        target_scaled_logits: {sid: log(p_y) - log(1 - p_y) on the target model}.
        shadow_out_signals: {sid: [scaled_logits from shadows where sid was NOT in
            the training set]}.
        fixed_variance: If True, use the median of per-sample sample-stds as a
            single global sigma (more stable for few shadows; default at ~10).
            If False, use per-sample sigma.
        min_out: minimum out-signals required to score a sample.
        sigma_floor: minimum effective sigma, prevents division by zero.

    Returns:
        (ids, scores) for samples present in both inputs with >= min_out shadow
        signals. Higher score = farther above the OUT distribution => more likely
        member.
    """
    valid_ids: List[int] = []
    mus: List[float] = []
    sigmas: List[float] = []
    targets: List[float] = []
    for sid in sorted(set(target_scaled_logits) & set(shadow_out_signals)):
        outs = np.asarray(shadow_out_signals[sid], dtype=np.float64)
        if outs.size < min_out:
            continue
        mus.append(float(outs.mean()))
        sigmas.append(float(outs.std(ddof=1)) if outs.size >= 2 else 0.0)
        targets.append(float(target_scaled_logits[sid]))
        valid_ids.append(int(sid))

    if not valid_ids:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    mus_a = np.asarray(mus)
    sigmas_a = np.asarray(sigmas)
    targets_a = np.asarray(targets)

    if fixed_variance:
        positive = sigmas_a[sigmas_a > 0]
        global_sigma = float(np.median(positive)) if positive.size else sigma_floor
        sigma_eff = np.full_like(sigmas_a, max(global_sigma, sigma_floor))
    else:
        sigma_eff = np.maximum(sigmas_a, sigma_floor)

    z = (targets_a - mus_a) / sigma_eff
    return np.asarray(valid_ids, dtype=np.int64), z.astype(np.float64)


def evaluate_threshold_attack(
    scores_member: np.ndarray,
    scores_nonmember: np.ndarray,
) -> ThresholdResult:
    """
    Compute AUC, balanced accuracy, and TPR at low FPR thresholds.

    Convention: higher score = more likely member. Non-finite scores (NaN/inf)
    are dropped before evaluation.
    """
    sm = np.asarray(scores_member, dtype=np.float64)
    snm = np.asarray(scores_nonmember, dtype=np.float64)
    sm = sm[np.isfinite(sm)]
    snm = snm[np.isfinite(snm)]

    if sm.size == 0 or snm.size == 0:
        raise ValueError(
            "Need at least one finite score per group "
            f"(members={sm.size}, non-members={snm.size})."
        )

    y = np.concatenate(
        [np.ones(sm.size, dtype=np.int64), np.zeros(snm.size, dtype=np.int64)]
    )
    s = np.concatenate([sm, snm])

    auc = float(roc_auc_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    bal_acc = float(np.max((tpr + (1.0 - fpr)) / 2.0))

    def tpr_at(fpr_target: float) -> float:
        ok = fpr <= fpr_target
        return float(tpr[ok].max()) if ok.any() else 0.0

    return ThresholdResult(
        auc=auc,
        balanced_accuracy=bal_acc,
        tpr_at_1pct_fpr=tpr_at(0.01),
        tpr_at_0_1pct_fpr=tpr_at(0.001),
        n_members=int(sm.size),
        n_nonmembers=int(snm.size),
    )
