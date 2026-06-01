"""
Phase D orchestrator: per-algorithm unlearning + per-round MIA.

For each unlearning algorithm:
  1. Copy contributions/<DATASET>/ to contributions/<DATASET>_UNLEARNED_<algo>/
  2. Apply unlearn_client_all_rounds(<client>, propagate=True) via manage_withdrawals.py
  3. Run evaluate_mia_per_round.py against the unlearned DB (target=unlearned)

Each algorithm gets its own DB copy so they don't interfere. Subsequent
plot_residual_map.py runs over the per-algorithm JSON outputs.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

PYTHON = sys.executable

# Tag -> (manage_withdrawals.py --algorithm value, extra CLI flags).
# Tags drive the per-algorithm DB-suffix and MIA output filename. The mapped
# "--algorithm value" is what manage_withdrawals.py sees, so multiple tagged
# configurations (e.g. class_pruning_p05_ft2) can share an underlying algorithm.
ALGO_CONFIGS = {
    "gradient":              ("gradient",           []),
    "influence":             ("influence",          ["--damping-factor", "0.01"]),
    "hessian":               ("hessian",            ["--damping-factor", "0.1", "--cg-max-iter", "30"]),
    "class_pruning":         ("class_pruning",      ["--prune-ratio", "0.1", "--n-probe-per-class", "64", "--finetune-epochs", "0"]),
    "class_pruning_p05_ft2": ("class_pruning",      ["--prune-ratio", "0.05", "--n-probe-per-class", "64", "--finetune-epochs", "2", "--finetune-lr", "1e-3"]),
    "gradient_ascent_kd":    ("gradient_ascent_kd", ["--ga-epochs", "3", "--alpha-retain", "1.0", "--gamma-kd", "1.0",
                                                     "--beta-forget", "1.0", "--kd-temperature", "4.0", "--finetune-lr", "1e-3"]),
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--dataset",
        default="MNIST",
        help=(
            "Canonical torchvision dataset name (MNIST/CIFAR10/FashionMNIST/CUSTOM). "
            "Used by the MIA driver for transforms/loaders. ALSO used as the source DB "
            "subdir unless --source-db-name overrides."
        ),
    )
    p.add_argument(
        "--source-db-name",
        default=None,
        help=(
            "Source contributions/ subdir to copy from (e.g. CIFAR10_DIRICHLET_a0p1). "
            "Defaults to --dataset for back-compat. Per-algo copies append _UNLEARNED_<algo>."
        ),
    )
    p.add_argument("--model", default="simplenet")
    p.add_argument("--num-clients", type=int, default=4)
    p.add_argument("--target-client", type=int, default=0, help="Client to unlearn AND target with MIA.")
    p.add_argument(
        "--algorithms",
        default=",".join(ALGO_CONFIGS.keys()),
        help="Comma-separated algorithm tags to run. Defaults to all configured tags.",
    )
    p.add_argument("--mia-attacks", default="loss,modent")
    p.add_argument("--mia-rounds", default="all")
    p.add_argument("--mia-n-members", type=int, default=1000)
    p.add_argument("--mia-n-nonmembers", type=int, default=1000)
    p.add_argument("--mia-non-member-source", default="test", choices=["test", "train_pool"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="If a per-algorithm DB already exists, reuse it (skip copy+unlearn).",
    )
    p.add_argument("--results-dir", default=".")
    return p.parse_args()


def copy_db(src: Path, dst: Path, skip_existing: bool) -> None:
    if dst.exists():
        if skip_existing:
            print(f"  [reuse] {dst} already exists, skipping copy")
            return
        print(f"  [clean] removing existing {dst}")
        shutil.rmtree(dst)
    print(f"  [copy ] {src} -> {dst}")
    shutil.copytree(src, dst)


def run(cmd: List[str]) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    args = parse_args()
    source_db_name = args.source_db_name or args.dataset
    src_db = Path("contributions") / source_db_name
    if not src_db.is_dir():
        sys.exit(f"Source DB not found: {src_db}")

    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    bad = [a for a in algos if a not in ALGO_CONFIGS]
    if bad:
        sys.exit(f"Unknown algorithms: {bad}. Available: {list(ALGO_CONFIGS)}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for algo in algos:
        print(f"\n{'='*70}\n[phase-d] algorithm: {algo}\n{'='*70}")
        t0 = time.monotonic()
        algo_db_name = f"{source_db_name}_UNLEARNED_{algo}"
        algo_db = Path("contributions") / algo_db_name
        unlearn_alg_name, unlearn_extra_flags = ALGO_CONFIGS[algo]

        copy_db(src_db, algo_db, args.skip_existing)

        unlearn_cmd = [
            PYTHON, "manage_withdrawals.py", "unlearn",
            "--dataset", algo_db_name,
            "--client-id", str(args.target_client),
            "--algorithm", unlearn_alg_name,
            "--num-clients", str(args.num_clients),
            "--model", args.model,
            "--no-evaluate",  # MIA gives us the privacy metric; skip the slower utility eval here
            *unlearn_extra_flags,
        ]
        rc = run(unlearn_cmd)
        if rc != 0:
            print(f"  [FAIL ] manage_withdrawals.py unlearn rc={rc}; skipping MIA for {algo}")
            summary.append((algo, "unlearn-failed", time.monotonic() - t0))
            continue

        mia_output = results_dir / f"mia_unlearned_{algo}.json"
        mia_plot = results_dir / f"mia_unlearned_{algo}.png"
        mia_cmd = [
            PYTHON, "evaluate_mia_per_round.py",
            "--db-dir", "contributions",
            "--dataset", args.dataset,         # canonical torchvision dataset
            "--db-name", algo_db_name,         # suffixed contributions subdir
            "--member-source-db-name", source_db_name,
            "--model", args.model,
            "--target-client", str(args.target_client),
            "--target", "both",  # both original and unlearned curves on the same plot
            "--attacks", args.mia_attacks,
            "--rounds", args.mia_rounds,
            "--n-members", str(args.mia_n_members),
            "--n-nonmembers", str(args.mia_n_nonmembers),
            "--non-member-source", args.mia_non_member_source,
            "--seed", str(args.seed),
            "--output", str(mia_output),
            "--plot", str(mia_plot),
        ]
        rc = run(mia_cmd)
        if rc != 0:
            print(f"  [FAIL ] evaluate_mia_per_round.py rc={rc}")
            summary.append((algo, "mia-failed", time.monotonic() - t0))
            continue

        elapsed = time.monotonic() - t0
        summary.append((algo, "ok", elapsed))
        print(f"  [done ] {algo} in {elapsed:.1f}s")

    print(f"\n{'='*70}\n[phase-d] summary\n{'='*70}")
    for algo, status, elapsed in summary:
        print(f"  {algo:20s}  {status:14s}  {elapsed:7.1f}s")


if __name__ == "__main__":
    main()
