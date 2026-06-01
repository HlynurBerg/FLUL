"""
Phase 1.2 driver — lower-bound K sweep K in {1,2,3} (D2.3 deferred bridge).

Sequencing:
  1. Delegate to run_phase_23.py --k-list 1,2,3 --algos class_pruning_p05_ft2,gradient
     to train the 6 new continuation DBs (CIFAR10_WD_K{01,02,03}_<algo>).
  2. Run evaluate_mia_per_round.py on each new DB, then read existing Phase 2.3
     JSONs for K in {5,10,15,20} to reuse those results.
  3. Build the augmented withdrawal-timing plot covering K in {1,2,3,5,10,15,20}.

Outputs:
  - 6 new continuation DBs under contributions/CIFAR10_WD_K{01,02,03}_<algo>/
  - mia_phase12_K{01,02,03}_{class_pruning_p05_ft2,gradient}.json
  - phase12_withdraw_timing_extended.png  (replacement for phase23_withdraw_timing.png)
  - phase12_summary.json
"""
from __future__ import annotations

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PYTHON = sys.executable

DEFAULT_DATASET = "CIFAR10"
DEFAULT_MODEL = "resnet18"
NEW_K_LIST = [1, 2, 3]
EXISTING_K_LIST = [5, 10, 15, 20]  # K=20 anchor reuses Phase 2.2 final-round MIA
ALGOS = ["class_pruning_p05_ft2", "gradient"]

# Phase 2.3 JSONs to read for K in {5,10,15}.
PHASE23_JSON = lambda k, algo: Path(f"mia_phase23_K{k:02d}_{algo}.json")
# K=20 anchor: reuse Phase 2.2 finals (mia_unlearned_<algo>.json at round 20).
PHASE22_JSON = {
    "class_pruning_p05_ft2": Path("mia_unlearned_class_pruning_p05_ft2.json"),
    "gradient": Path("mia_unlearned_gradient.json"),
}

# Reference floors / ceilings at round 20 (loss attack)
REFERENCE_RETRAIN_JSON = Path("mia_retrain_cifar10.json")
REFERENCE_ORIGINAL_JSON = Path("mia_original_cifar10.json")


def _new_db_name(dataset: str, k: int, algo: str) -> str:
    return f"{dataset}_WD_K{k:02d}_{algo}"


def _new_mia_json(k: int, algo: str) -> Path:
    return Path(f"mia_phase12_K{k:02d}_{algo}.json")


def _run_training(args) -> int:
    """Invoke run_phase_23.py to produce the 6 new K=1/2/3 continuation DBs."""
    # Skip if all 6 target DBs already exist (idempotent).
    if not args.force_train:
        all_exist = True
        for k in NEW_K_LIST:
            for algo in ALGOS:
                db = Path("contributions") / _new_db_name(args.dataset, k, algo)
                if not (db / f"round_{k:04d}").is_dir():
                    all_exist = False
                    break
            if not all_exist:
                break
        if all_exist:
            print("[phase12] All new K=1/2/3 continuation DBs already present — skipping training.")
            return 0

    cmd = [
        PYTHON, "run_phase_23.py",
        "--dataset", args.dataset,
        "--model", args.model,
        "--num-clients", "4",
        "--total-rounds", str(args.total_rounds),
        "--exclude-client", "0",
        "--k-list", ",".join(str(k) for k in NEW_K_LIST),
        "--algos", ",".join(ALGOS),
        "--log-dir", "logs/phase12",
        "--training-trace", "loss",
    ]
    print(f"[phase12] {' '.join(cmd)}")
    return subprocess.call(cmd)


def _run_mia_one(args, k: int, algo: str) -> Path:
    """Score per-round MIA on the new CIFAR10_WD_K0<K>_<algo> DB."""
    out_json = _new_mia_json(k, algo)
    if out_json.exists() and not args.force:
        print(f"[phase12] {out_json} exists — reusing.")
        return out_json
    cmd = [
        PYTHON, "evaluate_mia_per_round.py",
        "--db-dir", "contributions",
        "--dataset", args.dataset,
        "--db-name", _new_db_name(args.dataset, k, algo),
        "--member-source-db-name", args.dataset,
        "--model", args.model,
        "--target-client", "0",
        "--target", "original",
        "--attacks", "loss,modent",
        "--rounds", "all",
        "--n-members", "1000",
        "--n-nonmembers", "1000",
        "--non-member-source", "test",
        "--seed", str(args.seed),
        "--output", str(out_json),
    ]
    print(f"[phase12] K={k} algo={algo}: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"[phase12] MIA failed for K={k} algo={algo} (rc={rc})")
    return out_json


def _read_round_auc(json_path: Path, round_num: int, attack: str = "loss",
                    target: str = "original") -> float:
    if not json_path.exists():
        return float("nan")
    data = json.loads(json_path.read_text())
    rd = data.get("results", {}).get(str(round_num))
    if rd is None:
        return float("nan")
    mvn = rd.get(target, {}).get(attack, {}).get("members_vs_nonmembers")
    return float(mvn["auc"]) if mvn and "auc" in mvn else float("nan")


def _plot_extended(args, auc_by_k: Dict[str, Dict[int, float]],
                   retrain_floor: float, ceiling: float,
                   attack: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"class_pruning_p05_ft2": "tab:blue", "gradient": "tab:orange"}
    for algo, curve in auc_by_k.items():
        ks = sorted(curve.keys())
        ys = [curve[k] for k in ks]
        ax.plot(ks, ys, marker="o", linewidth=2.0,
                color=colors.get(algo, "tab:purple"), label=algo)

    ax.axhline(retrain_floor, color="tab:green", linestyle="--",
               linewidth=1.4, label=f"retrain floor ({retrain_floor:.3f})")
    ax.axhline(ceiling, color="tab:red", linestyle="--",
               linewidth=1.4, label=f"original-FL ceiling ({ceiling:.3f})")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Withdrawal round K")
    ax.set_ylabel(f"MIA AUC ({attack}) @ round 20")
    ax.set_title("Phase 1.2 — Extended withdrawal-timing curve (K∈{1,2,3,5,10,15,20})")
    ax.set_xticks(sorted({k for curve in auc_by_k.values() for k in curve}))
    ax.set_ylim(0.45, 0.80)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[phase12] Plot: {output}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--total-rounds", type=int, default=20,
                   help="Final round T (matches Phase 2.3 / Phase 1.1).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-plot", default="phase12_withdraw_timing_extended.png")
    p.add_argument("--output-summary", default="phase12_summary.json")
    p.add_argument("--force", action="store_true",
                   help="Re-score MIA even if JSONs exist.")
    p.add_argument("--force-train", action="store_true",
                   help="Re-train K=1/2/3 continuations even if their DBs exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate prerequisites and exit before running anything.")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[phase12] Sweep new K={NEW_K_LIST} × algos={ALGOS}, total_rounds={args.total_rounds}")

    # Pre-flight: source DBs (CIFAR10_UNLEARNED_<algo>) must exist
    for algo in ALGOS:
        for k in NEW_K_LIST:
            src = Path("contributions") / f"{args.dataset}_UNLEARNED_{algo}" / f"round_{k:04d}"
            if not src.is_dir():
                raise SystemExit(f"[phase12] Missing source: {src}. Phase 2.2 unlearning must "
                                 f"have produced per-round aggregates for {algo}.")

    if args.dry_run:
        print("[phase12] --dry-run: prerequisites OK, exiting.")
        return

    # 1. Train the 3 new K continuations × 2 algos = 6 DBs (~35 min × 2 = 70 min total).
    t0 = time.monotonic()
    rc = _run_training(args)
    if rc != 0:
        raise SystemExit(f"[phase12] run_phase_23.py failed with rc={rc}")
    print(f"[phase12] Training pass: {(time.monotonic()-t0)/60:.1f} min")

    # 2. Per-round MIA on the 6 new DBs
    t0 = time.monotonic()
    new_mia = {}
    for algo in ALGOS:
        for k in NEW_K_LIST:
            new_mia[(k, algo)] = _run_mia_one(args, k, algo)
    print(f"[phase12] MIA pass: {(time.monotonic()-t0)/60:.1f} min")

    # 3. Build the extended curve: round-20 AUC for K in {1,2,3,5,10,15,20}
    auc_loss: Dict[str, Dict[int, float]] = {algo: {} for algo in ALGOS}
    auc_modent: Dict[str, Dict[int, float]] = {algo: {} for algo in ALGOS}

    # New K (read round 20 from each new MIA JSON)
    for (k, algo), path in new_mia.items():
        auc_loss[algo][k] = _read_round_auc(path, 20, "loss")
        auc_modent[algo][k] = _read_round_auc(path, 20, "modent")

    # Existing K=5/10/15 (Phase 2.3 JSONs only have round 20 entry)
    for algo in ALGOS:
        for k in [5, 10, 15]:
            p23 = PHASE23_JSON(k, algo)
            auc_loss[algo][k] = _read_round_auc(p23, 20, "loss")
            auc_modent[algo][k] = _read_round_auc(p23, 20, "modent")

    # K=20 anchor (Phase 2.2 final-round MIA). The mia_unlearned_<algo>.json files
    # contain BOTH 'original' and 'unlearned' target curves; we want 'unlearned'
    # at round 20 (post-unlearning AUC), not 'original' (pre-unlearning ceiling).
    for algo in ALGOS:
        p22 = PHASE22_JSON.get(algo)
        if p22 and p22.exists():
            auc_loss[algo][20] = _read_round_auc(p22, 20, "loss", target="unlearned")
            auc_modent[algo][20] = _read_round_auc(p22, 20, "modent", target="unlearned")

    retrain_floor_loss = _read_round_auc(REFERENCE_RETRAIN_JSON, 20, "loss")
    ceiling_loss = _read_round_auc(REFERENCE_ORIGINAL_JSON, 20, "loss")
    retrain_floor_modent = _read_round_auc(REFERENCE_RETRAIN_JSON, 20, "modent")
    ceiling_modent = _read_round_auc(REFERENCE_ORIGINAL_JSON, 20, "modent")

    _plot_extended(args, auc_loss, retrain_floor_loss, ceiling_loss, "loss",
                   Path(args.output_plot))
    _plot_extended(args, auc_modent, retrain_floor_modent, ceiling_modent, "modent",
                   Path(str(args.output_plot).replace(".png", "_modent.png")))

    # Summary table
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "auc_by_K_loss": {algo: {str(k): v for k, v in sorted(d.items())}
                          for algo, d in auc_loss.items()},
        "auc_by_K_modent": {algo: {str(k): v for k, v in sorted(d.items())}
                            for algo, d in auc_modent.items()},
        "retrain_floor": {"loss": retrain_floor_loss, "modent": retrain_floor_modent},
        "original_ceiling": {"loss": ceiling_loss, "modent": ceiling_modent},
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2))
    print(f"[phase12] Summary: {args.output_summary}")

    print("\n[phase12] AUC @ round 20 by withdrawal round K (loss attack):")
    header = "  K     " + "  ".join(f"{algo:<26}" for algo in ALGOS)
    print(header)
    for k in sorted({k for d in auc_loss.values() for k in d}):
        row = f"  {k:>3}   " + "  ".join(
            f"{auc_loss[algo].get(k, float('nan')):<26.3f}" for algo in ALGOS
        )
        print(row)
    print(f"  retrain floor (loss): {retrain_floor_loss:.3f}")
    print(f"  original ceiling (loss): {ceiling_loss:.3f}")


if __name__ == "__main__":
    main()
