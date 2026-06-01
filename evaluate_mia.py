"""
CLI tool for Membership Inference Attack (MIA) evaluation against a saved
contributions DB.

Loads the latest (or specified) aggregate from a federated learning run,
constructs member / forgotten / non-member sets, runs requested attacks, and
writes a JSON report.

Example:
    python evaluate_mia.py --db-dir contributions/ --dataset MNIST \
        --target both --attacks loss,modent --output mia_results.json

--db-dir is the PARENT of the <DATASET>/ subdir (matches ContributionDB(base_dir=...)).
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from contributions_db import ContributionDB
from mia import (
    evaluate_threshold_attack,
    forward_with_ids,
    lira_offline_scores,
    loss_attack_scores,
    modified_entropy_attack_scores,
)
from mia_eval_sets import (
    build_evaluation_loader,
    collect_forgotten_sample_ids,
    collect_member_sample_ids,
    sample_test_set_non_members,
    sample_train_pool_non_members,
)
from mia_shadow import ShadowSpec, compute_shadow_signal_matrix, train_shadow_models
from model import create_model, set_parameters

_AVAILABLE_ATTACKS = ("loss", "modent", "lira")


def _resolve_round(db: ContributionDB, round_num: Optional[int]) -> int:
    if round_num is not None:
        return int(round_num)
    round_dirs = [d for d in Path(db.contributions_dir).glob("round_*") if d.is_dir()]
    if not round_dirs:
        raise FileNotFoundError(
            f"No round_* directories in {db.contributions_dir}. "
            "Hint: --db-dir should be the parent of the <DATASET>/ subdir."
        )
    return max(int(d.name.split("_")[1]) for d in round_dirs)


def _load_aggregate(db: ContributionDB, round_num: int, target: str):
    """
    target in {'original', 'unlearned'}. Returns numpy params list, or None.

    Convention (matches unlearning.py + run_sample_unlearning_demo.py): oldest
    aggregate file = original FL aggregate, newest = post-unlearning. Unlearning
    always saves a fresh aggregate AFTER the FL save, so mtime ordering is
    reliable. Returns None for ``unlearned`` if only one file exists (no
    unlearning has happened yet).
    """
    round_dir = Path(db.contributions_dir) / f"round_{round_num:04d}"
    files = sorted(
        round_dir.glob("round_*_aggregated_*_params.pkl"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return None
    if target == "original":
        path = files[0]
    elif target == "unlearned":
        if len(files) < 2:
            return None
        path = files[-1]
    else:
        raise ValueError(f"Unknown target '{target}'")
    with open(path, "rb") as f:
        return pickle.load(f)


def _subsample(rng: np.random.Generator, ids: List[int], n: int) -> List[int]:
    if n >= len(ids):
        return sorted(ids)
    return sorted(int(x) for x in rng.choice(np.asarray(ids), size=n, replace=False))


def _run_attacks(
    net,
    device,
    member_loader,
    forgotten_loader,
    nonmember_loader,
    attacks: List[str],
    *,
    shadow_out_signals: Optional[Dict[int, List[float]]] = None,
) -> Dict[str, dict]:
    """Forward each loader once; run all requested attacks."""
    member_data = forward_with_ids(net, member_loader, device) if member_loader else {}
    forgotten_data = (
        forward_with_ids(net, forgotten_loader, device) if forgotten_loader else {}
    )
    nonmember_data = forward_with_ids(net, nonmember_loader, device)

    threshold_fns = {
        "loss": loss_attack_scores,
        "modent": modified_entropy_attack_scores,
    }

    out: Dict[str, dict] = {}
    for atk in attacks:
        atk_out: Dict[str, dict] = {}
        if atk in threshold_fns:
            fn = threshold_fns[atk]
            _, nm_scores = fn(nonmember_data)
            if member_data:
                _, m_scores = fn(member_data)
                atk_out["members_vs_nonmembers"] = asdict(
                    evaluate_threshold_attack(m_scores, nm_scores)
                )
            if forgotten_data:
                _, f_scores = fn(forgotten_data)
                atk_out["forgotten_vs_nonmembers"] = asdict(
                    evaluate_threshold_attack(f_scores, nm_scores)
                )
        elif atk == "lira":
            if shadow_out_signals is None:
                raise RuntimeError(
                    "LiRA attack requested but no shadow_out_signals provided. "
                    "Train shadows before invoking _run_attacks."
                )
            target = {sid: d["scaled_logit"] for sid, d in nonmember_data.items()}
            target.update({sid: d["scaled_logit"] for sid, d in member_data.items()})
            target.update({sid: d["scaled_logit"] for sid, d in forgotten_data.items()})
            ids, scores = lira_offline_scores(target, shadow_out_signals)
            id_to_score = dict(zip(ids.tolist(), scores.tolist()))

            def _scores_for(group: Dict[int, dict]) -> np.ndarray:
                return np.asarray(
                    [id_to_score[sid] for sid in group if sid in id_to_score],
                    dtype=np.float64,
                )

            nm_lira = _scores_for(nonmember_data)
            if member_data:
                m_lira = _scores_for(member_data)
                if m_lira.size and nm_lira.size:
                    atk_out["members_vs_nonmembers"] = asdict(
                        evaluate_threshold_attack(m_lira, nm_lira)
                    )
            if forgotten_data:
                f_lira = _scores_for(forgotten_data)
                if f_lira.size and nm_lira.size:
                    atk_out["forgotten_vs_nonmembers"] = asdict(
                        evaluate_threshold_attack(f_lira, nm_lira)
                    )
        else:
            raise ValueError(
                f"Unknown attack '{atk}'. Available: {_AVAILABLE_ATTACKS}"
            )
        out[atk] = atk_out
    return out


def _print_summary(target: str, results: Dict[str, dict]) -> None:
    print(f"\n=== Target: {target} ===")
    for atk, atk_results in results.items():
        for setup, m in atk_results.items():
            print(
                f"  {atk:7s} {setup:30s} "
                f"AUC={m['auc']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  "
                f"TPR@1%FPR={m['tpr_at_1pct_fpr']:.3f}  "
                f"TPR@0.1%FPR={m['tpr_at_0_1pct_fpr']:.3f}  "
                f"(n_m={m['n_members']}, n_nm={m['n_nonmembers']})"
            )


def parse_args():
    p = argparse.ArgumentParser(
        description="Run MIA attacks against a saved FL contributions DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--db-dir",
        required=True,
        help="Parent of <DATASET>/ subdir (e.g. ./contributions). "
        "Must match the base_dir used when the DB was written.",
    )
    p.add_argument(
        "--dataset", default="MNIST", choices=["MNIST", "CIFAR10", "FASHIONMNIST"]
    )
    p.add_argument("--model", default="simplenet", choices=["simplenet", "resnet18"])
    p.add_argument(
        "--round", type=int, default=None, help="Round number (default: latest)"
    )
    p.add_argument(
        "--target", default="both", choices=["original", "unlearned", "both"]
    )
    p.add_argument(
        "--attacks",
        default="loss,modent",
        help=f"Comma-separated subset of {_AVAILABLE_ATTACKS}",
    )
    p.add_argument("--n-members", type=int, default=1000)
    p.add_argument("--n-nonmembers", type=int, default=1000)
    p.add_argument(
        "--non-member-source",
        default="test",
        choices=["test", "train_pool"],
        help="Sample non-members from held-out test split (default) or "
        "training pool minus members. LiRA requires train_pool.",
    )
    p.add_argument("--n-shadow", type=int, default=10, help="(LiRA) shadow count")
    p.add_argument("--shadow-epochs", type=int, default=5, help="(LiRA) epochs per shadow")
    p.add_argument("--shadow-cache", default="mia_cache/shadow", help="(LiRA) cache dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output", default="mia_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    unknown = [a for a in attacks if a not in _AVAILABLE_ATTACKS]
    if unknown:
        raise ValueError(f"Unknown attacks: {unknown}. Available: {_AVAILABLE_ATTACKS}")
    if "lira" in attacks and args.non_member_source != "train_pool":
        raise ValueError(
            "LiRA requires --non-member-source train_pool because shadow models "
            "only train on the canonical training split. Either drop 'lira' from "
            "--attacks or pass --non-member-source train_pool."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    db = ContributionDB(base_dir=args.db_dir, dataset_name=args.dataset)
    round_num = _resolve_round(db, args.round)

    print(f"DB:        {db.contributions_dir}")
    print(f"Dataset:   {args.dataset}  Model: {args.model}  Device: {device}")
    print(f"Round:     {round_num}")
    print(f"Attacks:   {attacks}")

    all_members = collect_member_sample_ids(db)
    forgotten = collect_forgotten_sample_ids(db)
    members_minus_forgotten = sorted(all_members - forgotten)
    forgotten_sorted = sorted(forgotten)
    members_eval = _subsample(rng, members_minus_forgotten, args.n_members)

    if args.non_member_source == "test":
        nonmember_ids = sample_test_set_non_members(
            args.dataset, args.n_nonmembers, seed=args.seed
        )
        nonmember_train_split = False
    else:
        nonmember_ids = sample_train_pool_non_members(
            args.dataset,
            exclude_ids=all_members,
            n=args.n_nonmembers,
            seed=args.seed,
        )
        nonmember_train_split = True

    print(
        f"Members:    {len(all_members)} total; {len(members_eval)} evaluated "
        f"({len(forgotten)} forgotten withheld from member set)"
    )
    print(f"Forgotten:  {len(forgotten_sorted)}")
    print(f"NonMembers: {len(nonmember_ids)} from {args.non_member_source} split")

    member_loader = (
        build_evaluation_loader(
            args.dataset, members_eval, train_split=True, batch_size=args.batch_size
        )
        if members_eval
        else None
    )
    forgotten_loader = (
        build_evaluation_loader(
            args.dataset,
            forgotten_sorted,
            train_split=True,
            batch_size=args.batch_size,
        )
        if forgotten_sorted
        else None
    )
    nonmember_loader = build_evaluation_loader(
        args.dataset,
        nonmember_ids,
        train_split=nonmember_train_split,
        batch_size=args.batch_size,
    )

    shadow_out_signals: Optional[Dict[int, List[float]]] = None
    shadow_spec_dict: Optional[dict] = None
    if "lira" in attacks:
        spec = ShadowSpec(
            dataset=args.dataset,
            model_name=args.model,
            n_shadow=args.n_shadow,
            sampling_rate=0.5,
            epochs=args.shadow_epochs,
            label_smoothing=0.0,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(
            f"\n[mia_shadow] Training/loading {args.n_shadow} shadow models for LiRA "
            f"(spec hash {spec.hash()}) into {args.shadow_cache}"
        )
        shadows = train_shadow_models(spec, args.shadow_cache, device)
        eval_ids = set(members_eval) | set(forgotten_sorted) | set(nonmember_ids)
        shadow_out_signals = compute_shadow_signal_matrix(shadows, eval_ids)
        shadow_spec_dict = {
            "hash": spec.hash(),
            "n_shadow": spec.n_shadow,
            "sampling_rate": spec.sampling_rate,
            "epochs": spec.epochs,
            "batch_size": spec.batch_size,
        }

    targets = ["original", "unlearned"] if args.target == "both" else [args.target]
    net = create_model(args.model, args.dataset).to(device)

    results: Dict[str, dict] = {}
    for target in targets:
        params = _load_aggregate(db, round_num, target)
        if params is None:
            print(f"\n[skip] No '{target}' aggregate found for round {round_num}")
            continue
        set_parameters(net, params)
        target_results = _run_attacks(
            net,
            device,
            member_loader,
            forgotten_loader,
            nonmember_loader,
            attacks,
            shadow_out_signals=shadow_out_signals,
        )
        results[target] = target_results
        _print_summary(target, target_results)

    output = {
        "dataset": args.dataset,
        "model": args.model,
        "round": round_num,
        "n_members_total": len(all_members),
        "n_members_evaluated": len(members_eval),
        "n_forgotten": len(forgotten_sorted),
        "n_nonmembers": len(nonmember_ids),
        "non_member_source": args.non_member_source,
        "attacks": attacks,
        "seed": args.seed,
        "shadow_spec": shadow_spec_dict,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
