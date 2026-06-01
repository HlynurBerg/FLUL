"""
Overlay-plotter for per-round MIA result JSONs.

Reads any number of JSON files produced by ``evaluate_mia_per_round.py`` and
draws the requested metric vs. round on a single axis. Intended for the
Phase 2.2 residual-map figure (original influence trace vs. each unlearning
algorithm's residual vs. retrained baseline) but agnostic to what the curves
actually mean — you label them.

Each curve is specified with ``--curve "label=path:target[:attack]"``:
  - label   — the legend entry
  - path    — JSON from evaluate_mia_per_round.py
  - target  — "original" or "unlearned" (key under each round's results)
  - attack  — "loss" / "modent" / "lira" (default: first attack listed in JSON)

Example (residual map):
    python plot_residual_map.py \
        --curve "Original=mia.json:original:loss" \
        --curve "Unlearned (influence)=mia_influence.json:unlearned:loss" \
        --curve "Unlearned (gradient_ascent_kd)=mia_kd.json:unlearned:loss" \
        --curve "Retrained baseline=mia_retrain.json:original:loss" \
        --output residual_map.png \
        --title "MNIST horizontal IID - client 0 leakage"
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_VALID_METRICS = (
    "auc",
    "balanced_accuracy",
    "tpr_at_1pct_fpr",
    "tpr_at_0_1pct_fpr",
)


@dataclass
class CurveSpec:
    label: str
    path: Path
    target: str
    attack: Optional[str]  # None → use first attack in the JSON


def _parse_curve(s: str) -> CurveSpec:
    """Parse "label=path:target[:attack]" into a CurveSpec."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--curve must be 'label=path:target[:attack]', got: {s!r}"
        )
    label, rest = s.split("=", 1)
    parts = rest.split(":")
    if len(parts) == 2:
        path, target = parts
        attack: Optional[str] = None
    elif len(parts) == 3:
        path, target, attack = parts
    else:
        raise argparse.ArgumentTypeError(
            f"--curve rhs must be 'path:target' or 'path:target:attack', got: {rest!r}"
        )
    if target not in ("original", "unlearned"):
        raise argparse.ArgumentTypeError(
            f"target must be 'original' or 'unlearned', got: {target!r}"
        )
    return CurveSpec(
        label=label.strip(),
        path=Path(path.strip()),
        target=target.strip(),
        attack=attack.strip() if attack else None,
    )


def _load_curve(spec: CurveSpec, metric: str) -> Tuple[List[int], List[float]]:
    """Pull (rounds, metric_values) for one curve from its JSON."""
    with open(spec.path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rounds: List[int] = list(data.get("rounds", []))
    if not rounds:
        # Fall back to whatever round keys are present in results
        rounds = sorted(int(r) for r in data.get("results", {}).keys())

    attack = spec.attack
    if attack is None:
        attacks = data.get("attacks", []) or []
        if not attacks:
            raise ValueError(f"{spec.path}: no 'attacks' field found")
        attack = attacks[0]

    results = data.get("results", {})
    ys: List[float] = []
    for r in rounds:
        cell = (
            results.get(str(r), {})
            .get(spec.target, {})
            .get(attack, {})
            .get("members_vs_nonmembers")
        )
        if cell is None or metric not in cell:
            ys.append(float("nan"))
        else:
            ys.append(float(cell[metric]))

    n_total = len(ys)
    n_finite = sum(1 for y in ys if y == y)  # NaN check
    if n_finite == 0:
        raise ValueError(
            f"{spec.path}: no {metric!r} values for target={spec.target!r} "
            f"attack={attack!r}. Available targets: {data.get('targets')}; "
            f"available attacks: {data.get('attacks')}"
        )
    if n_finite < n_total:
        print(
            f"[{spec.label}] {n_total - n_finite}/{n_total} rounds missing "
            f"{metric} for target={spec.target!r} attack={attack!r}"
        )
    return rounds, ys


def _metric_axis_label(metric: str) -> str:
    return {
        "auc": "MIA AUC (members vs non-members)",
        "balanced_accuracy": "MIA balanced accuracy",
        "tpr_at_1pct_fpr": "MIA TPR @ 1% FPR",
        "tpr_at_0_1pct_fpr": "MIA TPR @ 0.1% FPR",
    }.get(metric, metric)


def _plot(curves: List[Tuple[CurveSpec, List[int], List[float]]], args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    for spec, rounds, ys in curves:
        ax.plot(rounds, ys, marker="o", label=spec.label, linewidth=1.7)

    if args.metric == "auc" and args.show_random:
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="random=0.5")

    ax.set_xlabel("Round")
    ax.set_ylabel(_metric_axis_label(args.metric))
    if args.title:
        ax.set_title(args.title)
    if args.ylim:
        ax.set_ylim(*args.ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"\nPlot written to: {out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay per-round MIA curves from multiple JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--curve",
        action="append",
        required=True,
        type=_parse_curve,
        metavar="LABEL=PATH:TARGET[:ATTACK]",
        help="Repeatable. e.g. 'Original=mia.json:original:loss'.",
    )
    p.add_argument(
        "--metric",
        default="auc",
        choices=_VALID_METRICS,
        help="Which metric to plot from each round's members_vs_nonmembers cell.",
    )
    p.add_argument(
        "--output",
        default="residual_map.png",
        help="Output PNG path.",
    )
    p.add_argument("--title", default=None)
    p.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="y-axis range, e.g. --ylim 0.4 1.0.",
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[8.0, 5.0],
        metavar=("W", "H"),
    )
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument(
        "--show-random",
        action="store_true",
        default=True,
        help="Draw a horizontal y=0.5 reference line (only when --metric=auc).",
    )
    p.add_argument(
        "--no-show-random",
        action="store_false",
        dest="show_random",
        help="Suppress the y=0.5 reference line.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    curves = []
    for spec in args.curve:
        rounds, ys = _load_curve(spec, args.metric)
        curves.append((spec, rounds, ys))
        print(
            f"  {spec.label:40s}  {spec.path.name}  target={spec.target}  "
            f"attack={spec.attack or '(default)'}  rounds={len(rounds)}"
        )
    _plot(curves, args)


if __name__ == "__main__":
    main()
