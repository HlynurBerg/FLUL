"""
Tuned-config wrapper for class_pruning + gradient_ascent_kd on CIFAR-10 ResNet-18.

The default configs in run_phase_d.py (prune_ratio=0.1, beta_forget=1.0,
ga_epochs=3) destroy ResNet-18's utility on CIFAR-10. This script runs the same
two algorithms with milder configurations into suffixed DBs so the original
runs remain intact for comparison.

Suffix tag format: {dataset}_UNLEARNED_{algo}_{tag}

Output JSON / PNG: mia_unlearned_{algo}_{tag}.{json,png}
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable


# Each entry: (canonical_algo, tag, extra CLI flags for manage_withdrawals.py)
TUNED_CONFIGS = [
    ("class_pruning",      "p02",
     ["--prune-ratio", "0.02", "--n-probe-per-class", "64",
      "--finetune-epochs", "0"]),
    ("class_pruning",      "p05_ft2",
     ["--prune-ratio", "0.05", "--n-probe-per-class", "64",
      "--finetune-epochs", "2", "--finetune-lr", "1e-3"]),
    ("gradient_ascent_kd", "b03_e1",
     ["--ga-epochs", "1", "--alpha-retain", "1.0", "--gamma-kd", "1.0",
      "--beta-forget", "0.3", "--kd-temperature", "4.0", "--finetune-lr", "1e-3"]),
    ("gradient_ascent_kd", "b05_e2",
     ["--ga-epochs", "2", "--alpha-retain", "1.0", "--gamma-kd", "1.0",
      "--beta-forget", "0.5", "--kd-temperature", "4.0", "--finetune-lr", "1e-3"]),
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="CIFAR10")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--num-clients", type=int, default=4)
    p.add_argument("--target-client", type=int, default=0)
    p.add_argument("--mia-attacks", default="loss,modent")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-existing", action="store_true")
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


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    args = parse_args()
    src_db = Path("contributions") / args.dataset
    if not src_db.is_dir():
        sys.exit(f"Source DB not found: {src_db}")

    summary = []
    for algo, tag, flags in TUNED_CONFIGS:
        label = f"{algo}_{tag}"
        print(f"\n{'='*72}\n[tuned] {label}\n{'='*72}")
        t0 = time.monotonic()
        algo_db_name = f"{args.dataset}_UNLEARNED_{label}"
        algo_db = Path("contributions") / algo_db_name

        copy_db(src_db, algo_db, args.skip_existing)

        unlearn_cmd = [
            PYTHON, "manage_withdrawals.py", "unlearn",
            "--dataset", algo_db_name,
            "--client-id", str(args.target_client),
            "--algorithm", algo,
            "--num-clients", str(args.num_clients),
            "--model", args.model,
            "--no-evaluate",
            *flags,
        ]
        rc = run(unlearn_cmd)
        if rc != 0:
            summary.append((label, "unlearn-failed", time.monotonic() - t0))
            continue

        mia_output = f"mia_unlearned_{label}.json"
        mia_plot = f"mia_unlearned_{label}.png"
        mia_cmd = [
            PYTHON, "evaluate_mia_per_round.py",
            "--db-dir", "contributions",
            "--dataset", args.dataset,
            "--db-name", algo_db_name,
            "--model", args.model,
            "--target-client", str(args.target_client),
            "--target", "both",
            "--attacks", args.mia_attacks,
            "--rounds", "all",
            "--n-members", "1000",
            "--n-nonmembers", "1000",
            "--non-member-source", "test",
            "--seed", str(args.seed),
            "--output", mia_output,
            "--plot", mia_plot,
        ]
        rc = run(mia_cmd)
        if rc != 0:
            summary.append((label, "mia-failed", time.monotonic() - t0))
            continue
        elapsed = time.monotonic() - t0
        summary.append((label, "ok", elapsed))
        print(f"  [done ] {label} in {elapsed:.1f}s")

    print(f"\n{'='*72}\n[tuned] summary\n{'='*72}")
    for label, status, elapsed in summary:
        print(f"  {label:32s}  {status:14s}  {elapsed:7.1f}s")


if __name__ == "__main__":
    main()
