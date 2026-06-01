"""
Phase 1.1 driver — per-round resurfacing trace on existing IID continuation DBs.

For each (K, algo) in {5, 10, 15} x {class_pruning_p05_ft2, gradient}, run
evaluate_mia_per_round.py against contributions/CIFAR10_WD_K<K>_<algo>/ and
write a per-round JSON. Then build a 2x3 grid plot showing AUC vs round
for each cell, overlaid with the retrain floor (0.526) and original-FL ceiling
(0.733) references read from existing Phase 2.2 / 2.3 JSONs.

Re-uses existing artefacts when present (use --force to re-score).

Outputs:
  - mia_phase11_K{05,10,15}_{class_pruning_p05_ft2,gradient}.json  (6 JSONs)
  - phase11_resurfacing_trace.png                                  (2x3 grid)
  - phase11_summary.json                                           (consolidated)
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
from typing import Dict, List, Optional, Tuple

PYTHON = sys.executable

DEFAULT_DATASET = "CIFAR10"
DEFAULT_MODEL = "resnet18"
DEFAULT_K_LIST = [5, 10, 15]
DEFAULT_ALGOS = ["class_pruning_p05_ft2", "gradient"]

# Reference curves (Phase 2.2 originals)
REFERENCE_RETRAIN_JSON = "mia_retrain_cifar10.json"
REFERENCE_ORIGINAL_JSON = "mia_original_cifar10.json"


def _output_json(k: int, algo: str) -> Path:
    return Path(f"mia_phase11_K{k:02d}_{algo}.json")


def _wd_db_name(dataset: str, k: int, algo: str) -> str:
    return f"{dataset}_WD_K{k:02d}_{algo}"


def _verify_source(dataset: str, k: int, algo: str) -> Path:
    """Confirm contributions/<dataset>_WD_K<K>_<algo>/round_<K>/ exists."""
    db = Path("contributions") / _wd_db_name(dataset, k, algo)
    round_dir = db / f"round_{k:04d}"
    if not round_dir.is_dir():
        raise SystemExit(
            f"[phase11] Missing source DB: {round_dir}. Run Phase 2.3 first."
        )
    return db


def _run_mia(args, k: int, algo: str) -> Path:
    """Invoke evaluate_mia_per_round.py for one (K, algo) cell."""
    out_json = _output_json(k, algo)
    if out_json.exists() and not args.force:
        print(f"[phase11] {out_json} exists — reusing (pass --force to re-score)")
        return out_json
    db_name = _wd_db_name(args.dataset, k, algo)
    cmd = [
        PYTHON, "evaluate_mia_per_round.py",
        "--db-dir", "contributions",
        "--dataset", args.dataset,
        "--db-name", db_name,
        "--member-source-db-name", args.dataset,  # c0 sample IDs live in original
        "--model", args.model,
        "--target-client", str(args.target_client),
        "--target", "original",
        "--attacks", "loss,modent",
        "--rounds", "all",
        "--n-members", str(args.n_members),
        "--n-nonmembers", str(args.n_nonmembers),
        "--non-member-source", "test",
        "--seed", str(args.seed),
        "--output", str(out_json),
    ]
    print(f"[phase11] K={k} algo={algo}: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"[phase11] evaluate_mia_per_round.py failed for K={k} algo={algo} (rc={rc})")
    return out_json


def _read_per_round_auc(json_path: Path, attack: str) -> Dict[int, float]:
    """Returns {round: auc} for 'original' target / given attack."""
    data = json.loads(Path(json_path).read_text())
    out: Dict[int, float] = {}
    for r_str, rd in (data.get("results") or {}).items():
        mvn = (rd.get("original") or {}).get(attack, {}).get("members_vs_nonmembers")
        if mvn and "auc" in mvn:
            out[int(r_str)] = float(mvn["auc"])
    return out


def _read_reference(path: str, attack: str, target: str = "original") -> Dict[int, float]:
    """Read mia_original_cifar10.json / mia_retrain_cifar10.json style files."""
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    out: Dict[int, float] = {}
    for r_str, rd in (data.get("results") or {}).items():
        mvn = (rd.get(target) or {}).get(attack, {}).get("members_vs_nonmembers")
        if mvn and "auc" in mvn:
            out[int(r_str)] = float(mvn["auc"])
    return out


def _plot_grid(args, results_by_cell: Dict[Tuple[int, str], Dict[int, float]],
               retrain_curve: Dict[int, float], original_curve: Dict[int, float],
               attack: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_list = sorted({k for k, _ in results_by_cell})
    algos = args.algos

    fig, axes = plt.subplots(
        nrows=len(algos), ncols=len(k_list),
        figsize=(4.0 * len(k_list), 3.4 * len(algos)),
        sharex=False, sharey=True,
    )
    if len(algos) == 1:
        axes = [axes]
    if len(k_list) == 1:
        axes = [[ax] for ax in axes]

    retrain_floor = retrain_curve.get(20) if retrain_curve else None
    ceiling = original_curve.get(20) if original_curve else None

    for i, algo in enumerate(algos):
        for j, k in enumerate(k_list):
            ax = axes[i][j]
            curve = results_by_cell.get((k, algo), {})
            rs = sorted(curve.keys())
            ys = [curve[r] for r in rs]
            ax.plot(rs, ys, marker="o", linewidth=2, color="tab:blue", label="unlearned + cont.")
            # Reference curves over the same x-range
            if retrain_floor is not None:
                ax.axhline(retrain_floor, color="tab:green", linestyle="--",
                           linewidth=1.2, label=f"retrain floor ({retrain_floor:.3f})")
            if ceiling is not None:
                ax.axhline(ceiling, color="tab:red", linestyle="--",
                           linewidth=1.2, label=f"original-FL ceiling ({ceiling:.3f})")
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
            ax.set_xticks(rs)
            ax.set_xticklabels([str(r) for r in rs], fontsize=8)
            ax.set_title(f"K={k}  {algo}", fontsize=10)
            ax.set_ylim(0.45, 0.80)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"AUC ({attack})")
            if i == len(algos) - 1:
                ax.set_xlabel("Round")
            if i == 0 and j == len(k_list) - 1:
                ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(f"Phase 1.1 — Per-round resurfacing trace on CIFAR-10 ResNet-18 IID ({attack})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[phase11] Plot: {output}")


def _summarise(args, results: Dict[Tuple[int, str], Dict[int, float]],
               retrain_curve: Dict[int, float], original_curve: Dict[int, float],
               attack: str) -> Dict:
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "attack": attack,
        "retrain_floor_round20": retrain_curve.get(20),
        "original_ceiling_round20": original_curve.get(20),
        "cells": {},
    }
    print("\n[phase11] Loss-attack summary (round, AUC):")
    for (k, algo), curve in sorted(results.items()):
        rs = sorted(curve.keys())
        ys = [curve[r] for r in rs]
        cell_key = f"K{k:02d}_{algo}"
        summary["cells"][cell_key] = {
            "rounds": rs, "auc": ys,
            "min_auc": min(ys) if ys else None,
            "max_auc": max(ys) if ys else None,
            "endpoint_auc": ys[-1] if ys else None,
        }
        rng = f"{ys[0]:.3f}->{ys[-1]:.3f}" if ys else "--"
        print(f"  {cell_key:<36} rounds {rs[0]}..{rs[-1]}  AUC {rng}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--target-client", type=int, default=0)
    p.add_argument("--k-list", type=str, default=",".join(str(x) for x in DEFAULT_K_LIST))
    p.add_argument("--algos", type=str, default=",".join(DEFAULT_ALGOS))
    p.add_argument("--n-members", type=int, default=1000)
    p.add_argument("--n-nonmembers", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-plot", default="phase11_resurfacing_trace.png")
    p.add_argument("--output-summary", default="phase11_summary.json")
    p.add_argument("--force", action="store_true",
                   help="Re-score even if per-cell JSONs exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="Verify source DBs and exit before running MIA.")
    args = p.parse_args()
    args.k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    args.algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    return args


def main():
    args = parse_args()
    print(f"[phase11] Sweep: K={args.k_list} x algos={args.algos}")

    # Pre-flight
    for k in args.k_list:
        for algo in args.algos:
            _verify_source(args.dataset, k, algo)
    if args.dry_run:
        print("[phase11] --dry-run: source DBs verified, exiting.")
        return

    # MIA pass
    cells: Dict[Tuple[int, str], Path] = {}
    t0 = time.monotonic()
    for k in args.k_list:
        for algo in args.algos:
            cells[(k, algo)] = _run_mia(args, k, algo)
    dt = time.monotonic() - t0
    print(f"[phase11] MIA pass: {dt/60:.1f} min for {len(cells)} cells")

    # Collect results
    results_loss = {key: _read_per_round_auc(p, "loss") for key, p in cells.items()}
    results_modent = {key: _read_per_round_auc(p, "modent") for key, p in cells.items()}

    retrain_curve = _read_reference(REFERENCE_RETRAIN_JSON, "loss")
    original_curve = _read_reference(REFERENCE_ORIGINAL_JSON, "loss")

    # Plot loss (primary) and also save a modent companion plot
    _plot_grid(args, results_loss, retrain_curve, original_curve, "loss",
               Path(args.output_plot))
    modent_plot = Path(str(args.output_plot).replace(".png", "_modent.png"))
    _plot_grid(args, results_modent,
               _read_reference(REFERENCE_RETRAIN_JSON, "modent"),
               _read_reference(REFERENCE_ORIGINAL_JSON, "modent"),
               "modent", modent_plot)

    # Summary JSON
    summary_loss = _summarise(args, results_loss, retrain_curve, original_curve, "loss")
    summary_modent = _summarise(args, results_modent,
                                _read_reference(REFERENCE_RETRAIN_JSON, "modent"),
                                _read_reference(REFERENCE_ORIGINAL_JSON, "modent"),
                                "modent")
    summary = {"loss": summary_loss, "modent": summary_modent}
    Path(args.output_summary).write_text(json.dumps(summary, indent=2))
    print(f"[phase11] Summary: {args.output_summary}")


if __name__ == "__main__":
    main()
