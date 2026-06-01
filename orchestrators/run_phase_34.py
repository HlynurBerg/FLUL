"""
Phase 3.4 wrapper: cross-client contamination under non-IID.

For each Dirichlet alpha in --alpha-list, drive evaluate_mia_per_client.py on:
  - the BASELINE DB (no unlearning)            -> contributions/CIFAR10_DIRICHLET_a<alpha-tag>/
  - the post-unlearning DB for ALGO            -> contributions/CIFAR10_DIRICHLET_a<alpha-tag>_UNLEARNED_<algo>/

Scores client 1, 2, 3's LOCAL model artifacts at the final round against
client 0's training samples (members) vs. test-split non-members. Writes a
combined ``phase34_summary.json`` with per-condition c1/c2/c3 AUCs and the
delta vs. the matching aggregate-level AUC (read from the Phase 3.1 summary).

Example:
    python run_phase_34.py --dataset CIFAR10 --model resnet18 \\
        --alpha-list 0.5,0.1 --algo class_pruning_p05_ft2 \\
        --round 20 --evaluate-clients 1,2,3 --target-client 0 \\
        --phase31-summary phase31_summary.json \\
        --summary phase34_summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PYTHON = sys.executable


def _alpha_tag(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def _baseline_db_name(dataset: str, alpha: float) -> str:
    return f"{dataset}_DIRICHLET_a{_alpha_tag(alpha)}"


def _unlearned_db_name(dataset: str, alpha: float, algo: str) -> str:
    return f"{_baseline_db_name(dataset, alpha)}_UNLEARNED_{algo}"


def _run_per_client(args, db_name: str, mia_output: Path) -> int:
    cmd = [
        PYTHON, "evaluate_mia_per_client.py",
        "--db-dir", "contributions",
        "--dataset", args.dataset,
        "--db-name", db_name,
        "--member-source-db-name", _baseline_db_name(args.dataset, args.current_alpha),
        "--model", args.model,
        "--target-client", str(args.target_client),
        "--evaluate-clients", args.evaluate_clients,
        "--round", str(args.round),
        "--attacks", args.attacks,
        "--n-members", str(args.n_members),
        "--n-nonmembers", str(args.n_nonmembers),
        "--non-member-source", args.non_member_source,
        "--seed", str(args.seed),
        "--output", str(mia_output),
    ]
    print(f"[phase34] {' '.join(cmd)}")
    return subprocess.call(cmd)


def _extract_aggregate_auc(
    phase31: Optional[dict], alpha: float, algo: Optional[str], attack: str
) -> Optional[float]:
    """Pull the aggregate-level AUC (round 20) from the Phase 3.1 summary for
    one (alpha, algo) cell. ``algo=None`` returns the retrain baseline."""
    if not phase31:
        return None
    for pt in phase31.get("points", []):
        if abs(float(pt.get("point", -1)) - float(alpha)) > 1e-9:
            continue
        if algo is None:
            return (pt.get("retrain") or {}).get(attack)
        algo_block = (pt.get("algos") or {}).get(algo) or {}
        return algo_block.get(attack)
    return None


def _summarise(per_client: dict, attacks: List[str]) -> Dict[str, Dict[str, float]]:
    """Pivot per_client_results -> {attack: {client_id: auc, mean: ..., min: ..., max: ...}}."""
    out: Dict[str, Dict[str, float]] = {}
    for attack in attacks:
        per_client_aucs: Dict[str, float] = {}
        for cid_str, attack_block in (per_client.get("results") or {}).items():
            mvn = (attack_block.get(attack) or {}).get("members_vs_nonmembers") or {}
            if "auc" in mvn:
                per_client_aucs[cid_str] = float(mvn["auc"])
        if per_client_aucs:
            aucs = list(per_client_aucs.values())
            out[attack] = {
                **{f"c{cid}": v for cid, v in per_client_aucs.items()},
                "mean": sum(aucs) / len(aucs),
                "min": min(aucs),
                "max": max(aucs),
            }
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 3.4 - cross-client contamination under non-IID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "CIFAR10", "FashionMNIST"])
    p.add_argument("--model", default="resnet18")
    p.add_argument("--alpha-list", default="0.5,0.1",
                   help="Comma-separated Dirichlet alpha values from the Phase 3.1 sweep.")
    p.add_argument("--algo", default="class_pruning_p05_ft2",
                   help="Algorithm tag whose UNLEARNED DB should be scored alongside the baseline.")
    p.add_argument("--target-client", type=int, default=0)
    p.add_argument("--evaluate-clients", default="1,2,3")
    p.add_argument("--round", type=int, default=20)
    p.add_argument("--attacks", default="loss,modent")
    p.add_argument("--n-members", type=int, default=1000)
    p.add_argument("--n-nonmembers", type=int, default=1000)
    p.add_argument("--non-member-source", default="test", choices=["test", "train_pool"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phase31-summary", default=None,
                   help="Optional Phase 3.1 summary JSON; if present, the aggregate-level "
                        "AUC for the matching (alpha, algo) cell is included in the comparison.")
    p.add_argument("--results-dir", type=Path, default=Path("results/phase34"))
    p.add_argument("--summary", default="phase34_summary.json")
    return p.parse_args()


def main():
    args = parse_args()
    args.results_dir = Path(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(x) for x in args.alpha_list.split(",") if x.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]

    phase31 = None
    if args.phase31_summary and Path(args.phase31_summary).is_file():
        with open(args.phase31_summary, "r", encoding="utf-8") as f:
            phase31 = json.load(f)
        print(f"[phase34] Loaded aggregate context from {args.phase31_summary}")

    summary: Dict = {
        "dataset": args.dataset,
        "model": args.model,
        "target_client": args.target_client,
        "evaluate_clients": [int(x) for x in args.evaluate_clients.split(",") if x.strip()],
        "round": args.round,
        "attacks": attacks,
        "algo": args.algo,
        "alpha_list": alphas,
        "conditions": [],
    }

    for alpha in alphas:
        args.current_alpha = alpha
        tag = _alpha_tag(alpha)
        baseline_db = _baseline_db_name(args.dataset, alpha)
        unlearned_db = _unlearned_db_name(args.dataset, alpha, args.algo)

        # Condition 1: baseline (no unlearning)
        baseline_out = args.results_dir / f"mia_phase34_baseline_a{tag}.json"
        if not (Path("contributions") / baseline_db).is_dir():
            print(f"[phase34] SKIP alpha={alpha}: baseline DB missing ({baseline_db})")
            continue
        t0 = time.monotonic()
        rc = _run_per_client(args, baseline_db, baseline_out)
        baseline_block: Dict = {"alpha": alpha, "tag": tag, "type": "baseline",
                                 "db_name": baseline_db, "elapsed_seconds": time.monotonic() - t0,
                                 "rc": rc, "mia_output": str(baseline_out)}
        if rc == 0 and baseline_out.is_file():
            with open(baseline_out, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            baseline_block["per_client"] = _summarise(baseline_data, attacks)
            for attack in attacks:
                agg_auc = _extract_aggregate_auc(phase31, alpha, None, attack)  # retrain ref
                orig_agg_auc = _extract_aggregate_auc(phase31, alpha, None, attack)
                baseline_block.setdefault("aggregate_reference", {})[attack] = orig_agg_auc
        summary["conditions"].append(baseline_block)

        # Condition 2: unlearned (algo)
        unlearned_out = args.results_dir / f"mia_phase34_{args.algo}_a{tag}.json"
        if not (Path("contributions") / unlearned_db).is_dir():
            print(f"[phase34] SKIP alpha={alpha} unlearned: DB missing ({unlearned_db})")
            continue
        t0 = time.monotonic()
        rc = _run_per_client(args, unlearned_db, unlearned_out)
        unlearned_block: Dict = {"alpha": alpha, "tag": tag, "type": f"unlearned_{args.algo}",
                                  "db_name": unlearned_db, "elapsed_seconds": time.monotonic() - t0,
                                  "rc": rc, "mia_output": str(unlearned_out)}
        if rc == 0 and unlearned_out.is_file():
            with open(unlearned_out, "r", encoding="utf-8") as f:
                unlearned_data = json.load(f)
            unlearned_block["per_client"] = _summarise(unlearned_data, attacks)
            for attack in attacks:
                agg_auc = _extract_aggregate_auc(phase31, alpha, args.algo, attack)
                unlearned_block.setdefault("aggregate_reference", {})[attack] = agg_auc
        summary["conditions"].append(unlearned_block)

        # Incremental save
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"\n[phase34] Summary written to {args.summary}")


if __name__ == "__main__":
    main()
