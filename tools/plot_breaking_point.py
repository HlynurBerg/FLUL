"""
Plot the heterogeneity breaking-point curves from a Phase 3 summary JSON.

Reads ``phase31_summary.json`` (or ``phase32_summary.json``) produced by
``run_phase_31.py`` and draws ``gap_<attack>`` (y) vs. heterogeneity (x). One
line per algorithm. A horizontal reference line at ``--threshold`` marks the
"breaking point" criterion. Optionally emits a companion ``test_accuracy``
panel so the privacy gap is read alongside the utility cost.

x-axis depends on ``--mode``:
  - ``dirichlet``       - log-alpha from the summary's ``points[*].point`` (float).
  - ``class_vertical``  - primary share (linear) from ``points[*].point[0]``.

Example (Phase 3.1):
    python plot_breaking_point.py --summary phase31_summary.json \\
        --mode dirichlet --attack loss --threshold 0.05 \\
        --output phase31_breaking_point_loss.png

Example (Phase 3.2):
    python plot_breaking_point.py --summary phase32_summary.json \\
        --mode class_vertical --attack loss \\
        --output phase32_breaking_point_loss.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _x_value(mode: str, point) -> float:
    """Map a sweep point to its x-axis value."""
    if mode == "dirichlet":
        return float(point)
    # class_vertical: point is [primary, secondary] (JSON list)
    if isinstance(point, (list, tuple)) and len(point) >= 1:
        return float(point[0])
    return float(point)


def _x_label(mode: str) -> str:
    return "Dirichlet alpha (log scale)" if mode == "dirichlet" else "Primary class share"


def _series_per_algo(
    summary: dict, mode: str, attack: str
) -> Tuple[List[float], Dict[str, List[Optional[float]]], Dict[str, List[Optional[float]]],
           List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Pivot summary['points'] into:
      xs                 - x value per point
      gaps_per_algo      - algo -> [gap at each point] (None where missing)
      acc_per_algo       - algo -> [test_accuracy at each point]
      retrain_aucs       - [retrain MIA AUC at each point]
      orig_test_acc      - [original FL test acc at each point]
      retrain_test_acc   - [retrain test acc at each point]
    """
    xs: List[float] = []
    retrain_aucs: List[Optional[float]] = []
    orig_acc: List[Optional[float]] = []
    retrain_acc: List[Optional[float]] = []
    gaps: Dict[str, List[Optional[float]]] = {}
    accs: Dict[str, List[Optional[float]]] = {}

    for pt in summary.get("points", []):
        xs.append(_x_value(mode, pt.get("point")))
        retrain_aucs.append((pt.get("retrain") or {}).get(attack))
        orig_acc.append(pt.get("test_accuracy_original"))
        retrain_acc.append(pt.get("test_accuracy_retrain"))
        algos_block = pt.get("algos") or {}
        for algo, vals in algos_block.items():
            gaps.setdefault(algo, []).append(vals.get(f"gap_{attack}"))
            accs.setdefault(algo, []).append(vals.get("test_accuracy"))

    # Pad algos missing at some points so list lengths match xs.
    n = len(xs)
    for algo in list(gaps.keys()):
        while len(gaps[algo]) < n:
            gaps[algo].append(None)
        while len(accs.get(algo, [])) < n:
            accs.setdefault(algo, []).append(None)

    return xs, gaps, accs, retrain_aucs, orig_acc, retrain_acc


def _plot(args, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, gaps, accs, retrain_aucs, orig_acc, retrain_acc = _series_per_algo(
        summary, args.mode, args.attack
    )
    if not xs:
        raise SystemExit(f"No points found in {args.summary}")

    if args.with_accuracy:
        fig, (ax_gap, ax_acc) = plt.subplots(
            nrows=2, figsize=tuple(args.figsize), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
        )
    else:
        fig, ax_gap = plt.subplots(figsize=tuple(args.figsize))
        ax_acc = None

    # --- top panel: unlearning gap vs heterogeneity ---
    for algo, ys in gaps.items():
        x_plot = [x for x, y in zip(xs, ys) if y is not None]
        y_plot = [y for y in ys if y is not None]
        if x_plot:
            ax_gap.plot(x_plot, y_plot, marker="o", linewidth=1.7, label=algo)

    if args.threshold is not None:
        ax_gap.axhline(
            args.threshold,
            color="grey",
            linestyle="--",
            linewidth=0.9,
            label=f"breaking point = {args.threshold:.2f}",
        )
    ax_gap.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    ax_gap.set_ylabel(f"Unlearning gap (MIA AUC - retrain) [{args.attack}]")
    if args.title:
        ax_gap.set_title(args.title)
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="best", fontsize=9)

    if args.mode == "dirichlet":
        ax_gap.set_xscale("log")

    # --- bottom panel: test accuracy ---
    if ax_acc is not None:
        # original FL accuracy (no unlearning) and retrain accuracy as references.
        x_orig = [x for x, y in zip(xs, orig_acc) if y is not None]
        y_orig = [y for y in orig_acc if y is not None]
        if x_orig:
            ax_acc.plot(x_orig, y_orig, marker="s", linestyle="--", color="black",
                        linewidth=1.2, label="original FL")
        x_retr = [x for x, y in zip(xs, retrain_acc) if y is not None]
        y_retr = [y for y in retrain_acc if y is not None]
        if x_retr:
            ax_acc.plot(x_retr, y_retr, marker="^", linestyle="--", color="grey",
                        linewidth=1.2, label="retrain (no c0)")
        for algo, ys in accs.items():
            x_plot = [x for x, y in zip(xs, ys) if y is not None]
            y_plot = [y for y in ys if y is not None]
            if x_plot:
                ax_acc.plot(x_plot, y_plot, marker="o", linewidth=1.5, label=algo)
        ax_acc.set_ylabel("Test accuracy")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(loc="best", fontsize=9)
        ax_acc.set_xlabel(_x_label(args.mode))
        if args.mode == "dirichlet":
            ax_acc.set_xscale("log")
    else:
        ax_gap.set_xlabel(_x_label(args.mode))

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Plot written to: {out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot Phase 3 heterogeneity breaking-point curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--summary", type=Path, required=True, help="phase31_summary.json or phase32_summary.json")
    p.add_argument("--mode", choices=["dirichlet", "class_vertical"], required=True)
    p.add_argument("--attack", default="loss", choices=["loss", "modent", "lira"])
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Horizontal reference line for 'breaking point' gap. Set <0 to suppress.")
    p.add_argument("--output", type=Path, default=Path("phase3_breaking_point.png"))
    p.add_argument("--title", default=None)
    p.add_argument("--figsize", type=float, nargs=2, default=[8.0, 6.5], metavar=("W", "H"))
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--with-accuracy", action="store_true", default=True,
                   help="Include a companion test-accuracy panel below the gap panel.")
    p.add_argument("--no-with-accuracy", action="store_false", dest="with_accuracy",
                   help="Suppress the test-accuracy panel.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.threshold is not None and args.threshold < 0:
        args.threshold = None
    summary = _load_summary(args.summary)
    _plot(args, summary)


if __name__ == "__main__":
    main()
