"""
Per-client local-model MIA driver (Phase 2.4: cross-client contamination).

For a fixed (round, target_client) load every requested client's LOCAL model
artifact (``round_<R>/client_<C>/*_params.pkl``) and run MIA scoring the
target client's training samples as members vs. test-split non-members.

This isolates the *indirect-influence* channel: each remaining client received
the target-client-tainted aggregate and did one epoch of local SGD. The
question is whether their resulting local model still leaks the target's
membership signal.

Example:
    python evaluate_mia_per_client.py \\
        --db-dir contributions --dataset CIFAR10 --model resnet18 \\
        --db-name CIFAR10 --target-client 0 --evaluate-clients 1,2,3 \\
        --round 20 --attacks loss,modent --output mia_per_client_baseline.json

Notes:
- Client params have one *_params.pkl per (round, client) — we pick the OLDEST
  mtime to be robust if anything ever appended.
- ``--member-source-db-name`` lets you read the target's sample IDs from a
  different DB (e.g., the original FL run) when the scoring DB doesn't contain
  the target client at all (retrain baseline, or a Phase 2.3 continuation).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from contributions_db import ContributionDB
from evaluate_mia import _AVAILABLE_ATTACKS, _run_attacks, _subsample
from evaluate_mia_per_round import _collect_client_sample_ids
from mia_eval_sets import (
    build_evaluation_loader,
    collect_forgotten_sample_ids,
    sample_test_set_non_members,
    sample_train_pool_non_members,
)
from model import create_model, set_parameters


def _load_client_local_params(db: ContributionDB, round_num: int, client_id: int):
    """Load the OLDEST per-client *_params.pkl for (round, client). Returns None if missing."""
    client_dir = (
        Path(db.contributions_dir)
        / f"round_{round_num:04d}"
        / f"client_{client_id:04d}"
    )
    if not client_dir.is_dir():
        return None
    files = sorted(
        client_dir.glob("round_*_client_*_*_params.pkl"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return None
    with open(files[0], "rb") as f:
        return pickle.load(f)


def _resolve_round(db: ContributionDB, requested: Optional[int]) -> int:
    if requested is not None:
        return int(requested)
    dirs = [d for d in Path(db.contributions_dir).glob("round_*") if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(
            f"No round_* dirs in {db.contributions_dir}. "
            "Hint: --db-dir is the parent of the <DATASET>/ subdir."
        )
    return max(int(d.name.split("_")[1]) for d in dirs)


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-client local-model MIA driver (Phase 2.4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db-dir", required=True,
                   help="Parent of <DATASET>/ subdir (matches ContributionDB(base_dir=...)).")
    p.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "CIFAR10", "FASHIONMNIST"])
    p.add_argument("--db-name", default=None,
                   help="Contributions DB subdir. Defaults to --dataset.")
    p.add_argument("--member-source-db-name", default=None,
                   help="DB subdir to read target-client sample IDs from. Default = --db-name. "
                        "Use the original DB when scoring a continuation/retrain DB.")
    p.add_argument("--model", default="resnet18", choices=["simplenet", "resnet18"])
    p.add_argument("--target-client", type=int, default=0,
                   help="Client whose training samples are the MIA member set.")
    p.add_argument("--evaluate-clients", default="1,2,3",
                   help="Comma-separated list of client IDs whose local models to score.")
    p.add_argument("--round", type=int, default=None,
                   help="Round to load local models from. Default = latest round on disk.")
    p.add_argument("--attacks", default="loss,modent",
                   help=f"Subset of {_AVAILABLE_ATTACKS}. LiRA needs train_pool non-members.")
    p.add_argument("--n-members", type=int, default=1000)
    p.add_argument("--n-nonmembers", type=int, default=1000)
    p.add_argument("--non-member-source", default="test", choices=["test", "train_pool"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output", default="mia_per_client.json")
    p.add_argument(
        "--member-class-filter",
        type=int,
        default=None,
        help="If set, restrict the target client's member sample IDs to those "
        "whose canonical-train-split label equals this class index. Used by "
        "Phase 1.4 per-class breakdown.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    unknown = [a for a in attacks if a not in _AVAILABLE_ATTACKS]
    if unknown:
        raise ValueError(f"Unknown attacks: {unknown}. Available: {_AVAILABLE_ATTACKS}")
    if "lira" in attacks:
        raise SystemExit(
            "LiRA isn't wired into this per-client driver — shadow signals are "
            "uninformative on local-model copies. Use loss + modent."
        )
    evaluate_clients = [int(x) for x in args.evaluate_clients.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    db_name = args.db_name or args.dataset
    db = ContributionDB(base_dir=args.db_dir, dataset_name=db_name)
    member_source_db_name = args.member_source_db_name or db_name
    member_source_db = (
        db if member_source_db_name == db_name
        else ContributionDB(base_dir=args.db_dir, dataset_name=member_source_db_name)
    )
    round_num = _resolve_round(db, args.round)

    print(f"DB:                {db.contributions_dir}")
    print(f"Dataset/Model:     {args.dataset} / {args.model}  Device: {device}")
    print(f"Round:             {round_num}")
    print(f"Target client:     {args.target_client}")
    print(f"Evaluate clients:  {evaluate_clients}")
    print(f"Attacks:           {attacks}")

    client_members = _collect_client_sample_ids(member_source_db, args.target_client)
    if not client_members:
        raise RuntimeError(
            f"No sample manifests for client {args.target_client} in {member_source_db.contributions_dir}. "
            "If the scoring DB doesn't contain the target client, pass "
            "--member-source-db-name pointing at the original DB."
        )
    # Optional per-class filter (Phase 1.4): keep only target-client samples whose
    # canonical-train-split label matches member_class_filter.
    if args.member_class_filter is not None:
        from utils import _extract_targets
        from torch.utils.data import DataLoader
        # Load only labels — no images needed. We use load_data's transforms-aware
        # CIFAR-10 train dataset via _extract_targets, which reads ds.targets.
        from torchvision import datasets, transforms
        if args.dataset == "CIFAR10":
            ds = datasets.CIFAR10(root="./data", train=True, download=False,
                                  transform=transforms.ToTensor())
        elif args.dataset == "MNIST":
            ds = datasets.MNIST(root="./data", train=True, download=False,
                                transform=transforms.ToTensor())
        elif args.dataset == "FASHIONMNIST":
            ds = datasets.FashionMNIST(root="./data", train=True, download=False,
                                       transform=transforms.ToTensor())
        else:
            raise SystemExit(f"--member-class-filter not supported for dataset={args.dataset}")
        targets = _extract_targets(ds)
        k_filter = int(args.member_class_filter)
        before = len(client_members)
        client_members = {sid for sid in client_members
                          if 0 <= sid < len(targets) and int(targets[sid]) == k_filter}
        print(f"[filter] member_class_filter={k_filter}: kept {len(client_members)}/{before} "
              f"target-client samples in class {k_filter}.")
        if not client_members:
            raise RuntimeError(
                f"member-class-filter={k_filter} left zero members. "
                f"Check that target_client {args.target_client} has any class-{k_filter} samples."
            )
    forgotten = collect_forgotten_sample_ids(db)
    members_minus_forgotten = sorted(client_members - forgotten)
    members_eval = _subsample(rng, members_minus_forgotten, args.n_members)

    if args.non_member_source == "test":
        nonmember_ids = sample_test_set_non_members(
            args.dataset, args.n_nonmembers, seed=args.seed
        )
        nonmember_train_split = False
    else:
        nonmember_ids = sample_train_pool_non_members(
            args.dataset, exclude_ids=client_members, n=args.n_nonmembers, seed=args.seed
        )
        nonmember_train_split = True

    print(f"Members:    {len(client_members)} for client {args.target_client}; "
          f"{len(members_eval)} evaluated ({len(forgotten)} forgotten withheld)")
    print(f"NonMembers: {len(nonmember_ids)} from {args.non_member_source} split")

    member_loader = build_evaluation_loader(
        args.dataset, members_eval, train_split=True, batch_size=args.batch_size
    )
    nonmember_loader = build_evaluation_loader(
        args.dataset, nonmember_ids, train_split=nonmember_train_split, batch_size=args.batch_size
    )

    net = create_model(args.model, args.dataset).to(device)
    per_client_results: Dict[str, dict] = {}
    skipped: List[int] = []

    for cid in evaluate_clients:
        params = _load_client_local_params(db, round_num, cid)
        if params is None:
            print(f"  client {cid:4d}  no local-model artifact at round {round_num} — SKIP")
            skipped.append(cid)
            continue
        set_parameters(net, params)
        attack_results = _run_attacks(
            net, device, member_loader, None, nonmember_loader, attacks,
        )
        per_client_results[str(cid)] = attack_results
        primary = attacks[0]
        mvn = attack_results.get(primary, {}).get("members_vs_nonmembers", {})
        if "auc" in mvn:
            print(f"  client {cid:4d}  {primary} AUC={mvn['auc']:.3f}  "
                  f"bal_acc={mvn.get('balanced_accuracy', float('nan')):.3f}")

    if skipped:
        print(f"[skip] No local-model artifact at round {round_num} for clients: {skipped}")

    output = {
        "dataset": args.dataset,
        "model": args.model,
        "db_name": db_name,
        "member_source_db_name": member_source_db_name,
        "round": round_num,
        "target_client": args.target_client,
        "evaluate_clients": evaluate_clients,
        "attacks": attacks,
        "n_members_total": len(client_members),
        "n_members_evaluated": len(members_eval),
        "n_forgotten_excluded": len(forgotten),
        "n_nonmembers": len(nonmember_ids),
        "non_member_source": args.non_member_source,
        "seed": args.seed,
        "member_class_filter": args.member_class_filter,
        "results": per_client_results,
        "skipped_clients": skipped,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
