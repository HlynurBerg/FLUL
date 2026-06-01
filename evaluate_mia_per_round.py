"""
Per-round MIA driver. Loads the saved aggregate at each round of a federated
training run and runs the requested membership-inference attacks against a
target client's data.

This is the driver that produces:
  - Phase 2.1's baseline influence trace (target=original)
  - Phase 2.2's post-unlearning residual map (target=unlearned, on a DB that
    has had unlearn_client_all_rounds(propagate=True) applied)

Member sourcing differs from evaluate_mia.py: members are restricted to a
single target client's training samples (across all rounds it participated in)
so the resulting curve isolates that client's leakage signal.

Example:
    python evaluate_mia_per_round.py \
        --db-dir contributions/ --dataset MNIST --model simplenet \
        --target-client 0 --target original --attacks loss,modent \
        --output mia_per_round.json --plot mia_per_round.png

LiRA is supported but expensive: shadows are trained once and reused across
rounds (same dataset/model). Use --attacks lira --non-member-source train_pool.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from contributions_db import ContributionDB
from evaluate_mia import (
    _AVAILABLE_ATTACKS,
    _load_aggregate,
    _run_attacks,
    _subsample,
)
from mia_eval_sets import (
    build_evaluation_loader,
    collect_forgotten_sample_ids,
    sample_test_set_non_members,
    sample_train_pool_non_members,
)
from mia_shadow import ShadowSpec, compute_shadow_signal_matrix, train_shadow_models
from model import create_model, set_parameters


def _collect_client_sample_ids(db: ContributionDB, client_id: int) -> Set[int]:
    """Union of sample IDs across every round this client participated in.

    Mirrors mia_eval_sets.collect_member_sample_ids but filters to a single
    client subdir. Used so the per-round influence trace tracks ONE client's
    leakage signal rather than the federation-wide member set.
    """
    members: Set[int] = set()
    contrib_dir = Path(db.contributions_dir)
    client_subdir = f"client_{client_id:04d}"
    for round_dir in sorted(contrib_dir.glob("round_*")):
        if not round_dir.is_dir():
            continue
        client_dir = round_dir / client_subdir
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


def _enumerate_rounds(db: ContributionDB, rounds_arg: Optional[str]) -> List[int]:
    """Return the round numbers to evaluate. Default: every round_* dir on disk."""
    contrib_dir = Path(db.contributions_dir)
    available = sorted(
        int(d.name.split("_")[1])
        for d in contrib_dir.glob("round_*")
        if d.is_dir()
    )
    if not available:
        raise FileNotFoundError(
            f"No round_* directories under {contrib_dir}. "
            "Hint: --db-dir should be the parent of the <DATASET>/ subdir."
        )
    if rounds_arg in (None, "all", ""):
        return available
    requested = [int(x) for x in rounds_arg.split(",") if x.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Requested rounds not present in DB: {missing}")
    return requested


def _maybe_plot(
    output_path: str,
    rounds: List[int],
    targets: List[str],
    attacks: List[str],
    results: Dict[str, dict],
) -> None:
    """Plot per-round AUC for each (target, attack) pair. Members vs non-members."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[plot] matplotlib not available, skipping {output_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for target in targets:
        for atk in attacks:
            ys = []
            for r in rounds:
                m = (
                    results.get(str(r), {})
                    .get(target, {})
                    .get(atk, {})
                    .get("members_vs_nonmembers")
                )
                ys.append(m["auc"] if m else float("nan"))
            label = f"{target}/{atk}"
            ax.plot(rounds, ys, marker="o", label=label)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="random=0.5")
    ax.set_xlabel("Round")
    ax.set_ylabel("MIA AUC (members vs non-members)")
    ax.set_title("Per-round membership inference AUC")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Plot written to: {out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-round MIA driver: AUC vs training round for a target client.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--db-dir",
        required=True,
        help="Parent of <DATASET>/ subdir (matches ContributionDB(base_dir=...)).",
    )
    p.add_argument(
        "--dataset", default="MNIST", choices=["MNIST", "CIFAR10", "FASHIONMNIST"],
        help="Canonical torchvision dataset (used for member/non-member loaders + transforms).",
    )
    p.add_argument(
        "--db-name",
        default=None,
        help="Contributions DB subdir name. Defaults to --dataset. Use this when "
        "the DB was written with a partition or retrain suffix (e.g. "
        "'MNIST_RETRAINED_no0', 'MNIST_DIRICHLET_a0p1').",
    )
    p.add_argument(
        "--member-source-db-name",
        default=None,
        help="DB subdir to read the target client's sample IDs from. Defaults to "
        "--db-name. Override this when scoring a retrained baseline (where the "
        "target client wasn't part of training): point at the ORIGINAL DB so we "
        "still know which samples were 'members' in the original run.",
    )
    p.add_argument("--model", default="simplenet", choices=["simplenet", "resnet18"])
    p.add_argument(
        "--target-client",
        type=int,
        default=0,
        help="Client whose samples are the MIA member set.",
    )
    p.add_argument(
        "--rounds",
        default="all",
        help="Comma-separated round numbers, or 'all' for every round on disk.",
    )
    p.add_argument(
        "--target",
        default="original",
        choices=["original", "unlearned", "both"],
        help="Which aggregate to score. 'unlearned' requires propagation to have written per-round corrected aggregates.",
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
        help="LiRA requires train_pool.",
    )
    p.add_argument("--n-shadow", type=int, default=10, help="(LiRA) shadow count")
    p.add_argument("--shadow-epochs", type=int, default=5, help="(LiRA) epochs per shadow")
    p.add_argument("--shadow-cache", default="mia_cache/shadow", help="(LiRA) cache dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output", default="mia_per_round.json")
    p.add_argument(
        "--plot",
        default=None,
        help="Optional path for an AUC-vs-round PNG. Omit to skip plotting.",
    )
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
            "only train on the canonical training split."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    db_name = args.db_name or args.dataset
    db = ContributionDB(base_dir=args.db_dir, dataset_name=db_name)
    member_source_db_name = args.member_source_db_name or db_name
    member_source_db = (
        db
        if member_source_db_name == db_name
        else ContributionDB(base_dir=args.db_dir, dataset_name=member_source_db_name)
    )
    rounds = _enumerate_rounds(db, args.rounds)
    targets = ["original", "unlearned"] if args.target == "both" else [args.target]

    print(f"DB:            {db.contributions_dir}")
    print(f"Dataset/Model: {args.dataset} / {args.model}  Device: {device}")
    print(f"Target client: {args.target_client}")
    print(f"Rounds:        {rounds[0]}..{rounds[-1]} ({len(rounds)} total)")
    print(f"Targets:       {targets}")
    print(f"Attacks:       {attacks}")

    client_members = _collect_client_sample_ids(member_source_db, args.target_client)
    if not client_members:
        raise RuntimeError(
            f"Found no sample manifests for client {args.target_client} in "
            f"{member_source_db.contributions_dir}. Did training write per-sample "
            "manifests (--training-trace loss|full)? If this is a retrained "
            "baseline DB, pass --member-source-db-name pointing at the original DB."
        )
    # Forgotten withdrawals are a property of the SCORING DB (post-unlearning),
    # not the member-source DB. Read from the scoring DB.
    forgotten = collect_forgotten_sample_ids(db)
    members_minus_forgotten = sorted(client_members - forgotten)
    if not members_minus_forgotten:
        raise RuntimeError(
            f"All of client {args.target_client}'s samples are marked withdrawn — nothing left to score."
        )
    members_eval = _subsample(rng, members_minus_forgotten, args.n_members)

    if args.non_member_source == "test":
        nonmember_ids = sample_test_set_non_members(
            args.dataset, args.n_nonmembers, seed=args.seed
        )
        nonmember_train_split = False
    else:
        nonmember_ids = sample_train_pool_non_members(
            args.dataset,
            exclude_ids=client_members,
            n=args.n_nonmembers,
            seed=args.seed,
        )
        nonmember_train_split = True

    print(
        f"Members:    {len(client_members)} for client {args.target_client}; "
        f"{len(members_eval)} evaluated ({len(forgotten)} forgotten withheld)"
    )
    print(f"NonMembers: {len(nonmember_ids)} from {args.non_member_source} split")

    member_loader = build_evaluation_loader(
        args.dataset, members_eval, train_split=True, batch_size=args.batch_size
    )
    nonmember_loader = build_evaluation_loader(
        args.dataset,
        nonmember_ids,
        train_split=nonmember_train_split,
        batch_size=args.batch_size,
    )

    shadow_out_signals: Optional[Dict[int, list]] = None
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
            f"\n[mia_shadow] Training/loading {args.n_shadow} shadow models "
            f"(spec hash {spec.hash()}) into {args.shadow_cache}"
        )
        shadows = train_shadow_models(spec, args.shadow_cache, device)
        eval_ids = set(members_eval) | set(nonmember_ids)
        shadow_out_signals = compute_shadow_signal_matrix(shadows, eval_ids)
        shadow_spec_dict = {
            "hash": spec.hash(),
            "n_shadow": spec.n_shadow,
            "sampling_rate": spec.sampling_rate,
            "epochs": spec.epochs,
            "batch_size": spec.batch_size,
        }

    net = create_model(args.model, args.dataset).to(device)
    per_round_results: Dict[str, dict] = {}
    skipped: Dict[str, list] = {t: [] for t in targets}

    for r in rounds:
        per_target: Dict[str, dict] = {}
        for target in targets:
            params = _load_aggregate(db, r, target)
            if params is None:
                skipped[target].append(r)
                continue
            set_parameters(net, params)
            target_results = _run_attacks(
                net,
                device,
                member_loader,
                None,  # no separate "forgotten" stream — the script focuses on members
                nonmember_loader,
                attacks,
                shadow_out_signals=shadow_out_signals,
            )
            per_target[target] = target_results
            mvn = (
                target_results.get(attacks[0], {}).get("members_vs_nonmembers", {})
            )
            auc = mvn.get("auc")
            if auc is not None:
                print(f"  round {r:4d}  {target:9s}  {attacks[0]} AUC={auc:.3f}")
        if per_target:
            per_round_results[str(r)] = per_target

    for target, missing in skipped.items():
        if missing:
            print(f"[skip] No '{target}' aggregate at rounds: {missing}")

    output = {
        "dataset": args.dataset,
        "model": args.model,
        "target_client": args.target_client,
        "rounds": rounds,
        "targets": targets,
        "attacks": attacks,
        "n_members_total": len(client_members),
        "n_members_evaluated": len(members_eval),
        "n_forgotten_excluded": len(forgotten),
        "n_nonmembers": len(nonmember_ids),
        "non_member_source": args.non_member_source,
        "seed": args.seed,
        "shadow_spec": shadow_spec_dict,
        "results": per_round_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {out_path}")

    if args.plot:
        _maybe_plot(args.plot, rounds, targets, attacks, per_round_results)


if __name__ == "__main__":
    main()
