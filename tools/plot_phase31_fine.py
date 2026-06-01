"""Plot the Phase 3.1 alpha sweep on the fine grid, merging the original
coarse summary with the finer-sweep follow-up summaries.

Reads:
  - phase31_summary.json                (original grid: 1000, 10, 1, 0.5, 0.1, 0.05)
  - phase31_fine_summary.json           (fine grid: 0.2, 0.3, 0.4 + partial 0.6)
  - phase31_a0p7_summary.json           (a0p7 full)
  - phase31_a0p8_summary.json           (a0p8 full)
  - results/phase31/point_a0p6/mia_unlearned_{influence,class_pruning_p05_ft2}.json
                                        (a0p6 algos missing from phase31_fine_summary)
  - results/phase31/mia_phase31_a0p6_retrain.json
                                        (a0p6 retrain baseline)

Emits one figure per attack (loss, modent) showing the unlearning gap vs.
Dirichlet alpha for all three algorithms, with fine-sweep points marked
distinctly from the original-grid points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
ORIGINAL_ALPHAS = {1000.0, 10.0, 1.0, 0.5, 0.1, 0.05}


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_auc(json_path: Path, target: str, attack: str, round_num: int = 20) -> Optional[float]:
    if not json_path.is_file():
        return None
    d = _load(json_path)
    block = d.get("results", {}).get(str(round_num), {}).get(target, {})
    return block.get(attack, {}).get("members_vs_nonmembers", {}).get("auc")


def _patch_a0p6(point: dict) -> dict:
    """phase31_fine_summary stored a0p6 with influence+class_pruning null
    because the first run crashed mid-phase-D. Pull the actual numbers
    from the per-point MIA JSONs written by the resume runs."""
    if point.get("tag") != "a0p6":
        return point
    a0p6_retrain_path = ROOT / "results" / "phase31" / "mia_phase31_a0p6_retrain.json"
    point_dir = ROOT / "results" / "phase31" / "point_a0p6"

    retrains = point.get("retrain") or {}
    for attack in ("loss", "modent"):
        if retrains.get(attack) is None:
            v = _extract_auc(a0p6_retrain_path, "original", attack)
            if v is not None:
                retrains[attack] = v
    point["retrain"] = retrains

    algos = point.get("algos") or {}
    for algo in ("influence", "class_pruning_p05_ft2"):
        block = algos.get(algo) or {}
        for attack in ("loss", "modent"):
            if block.get(attack) is None:
                v = _extract_auc(point_dir / f"mia_unlearned_{algo}.json", "unlearned", attack)
                if v is not None:
                    block[attack] = v
            r = retrains.get(attack)
            u = block.get(attack)
            if r is not None and u is not None:
                block[f"gap_{attack}"] = u - r
        algos[algo] = block
    point["algos"] = algos
    return point


def _merge_points(summaries: List[dict]) -> List[dict]:
    """Concatenate all points, patch a0p6, then sort by alpha ascending."""
    out: List[dict] = []
    for s in summaries:
        for pt in s.get("points", []):
            out.append(_patch_a0p6(pt))
    out.sort(key=lambda p: float(p.get("point", 0.0)))
    return out


def _series(points: List[dict], attack: str, algos: List[str]) \
        -> Tuple[List[float], Dict[str, List[Optional[float]]], List[bool]]:
    xs = [float(p["point"]) for p in points]
    is_new = [float(p["point"]) not in ORIGINAL_ALPHAS for p in points]
    gaps: Dict[str, List[Optional[float]]] = {a: [] for a in algos}
    for p in points:
        algo_block = p.get("algos") or {}
        for a in algos:
            v = (algo_block.get(a) or {}).get(f"gap_{attack}")
            gaps[a].append(v)
    return xs, gaps, is_new


ALGO_LABEL = {
    "gradient": "gradient",
    "influence": "influence",
    "class_pruning_p05_ft2": "class_pruning_p05_ft2",
}
ALGO_COLOR = {
    "gradient": "#1f77b4",
    "influence": "#d62728",
    "class_pruning_p05_ft2": "#2ca02c",
}


def _plot(points: List[dict], attack: str, out_path: Path,
          threshold: Optional[float] = 0.05) -> None:
    algos = ["gradient", "influence", "class_pruning_p05_ft2"]
    xs, gaps, is_new = _series(points, attack, algos)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for algo in algos:
        ys = gaps[algo]
        # Connect all valid points with one line per algo.
        x_line = [x for x, y in zip(xs, ys) if y is not None]
        y_line = [y for y in ys if y is not None]
        ax.plot(x_line, y_line, "-", color=ALGO_COLOR[algo], linewidth=1.7,
                alpha=0.75, zorder=2)

        # Original-grid markers: filled circles.
        x_orig = [x for x, y, n in zip(xs, ys, is_new) if y is not None and not n]
        y_orig = [y for y, n in zip(ys, is_new) if y is not None and not n]
        ax.plot(x_orig, y_orig, "o", color=ALGO_COLOR[algo], markersize=7,
                markeredgecolor="white", markeredgewidth=0.8,
                label=ALGO_LABEL[algo], zorder=3)

        # Fine-sweep markers: open diamonds.
        x_new = [x for x, y, n in zip(xs, ys, is_new) if y is not None and n]
        y_new = [y for y, n in zip(ys, is_new) if y is not None and n]
        ax.plot(x_new, y_new, "D", markerfacecolor="none",
                markeredgecolor=ALGO_COLOR[algo], markeredgewidth=1.4,
                markersize=7, zorder=3)

    # Reference horizontal lines.
    if threshold is not None:
        ax.axhline(threshold, color="grey", linestyle="--", linewidth=0.9,
                   label=f"breaking point = {threshold:.2f}")
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)

    # Marker legend (filled vs open).
    from matplotlib.lines import Line2D
    extra_handles = [
        Line2D([0], [0], marker="o", color="black", markersize=7,
               markerfacecolor="black", markeredgecolor="white",
               linestyle="None", label="original grid"),
        Line2D([0], [0], marker="D", color="black", markersize=7,
               markerfacecolor="none", markeredgecolor="black",
               markeredgewidth=1.4, linestyle="None", label="finer-sweep follow-up"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(extra_handles)
    labels.extend([h.get_label() for h in extra_handles])
    ax.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.92)

    ax.set_xscale("log")
    ax.set_xlabel(r"Dirichlet $\alpha$ (log scale)")
    ax.set_ylabel(f"Unlearning gap (MIA AUC $-$ retrain) [{attack}]")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Plot written to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "new_report" / "Images",
                   help="Where to write phase31_breaking_point_fine_{loss,modent}.png")
    p.add_argument("--threshold", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    summaries = [
        _load(ROOT / "phase31_summary.json"),
        _load(ROOT / "phase31_fine_summary.json"),
        _load(ROOT / "phase31_a0p7_summary.json"),
        _load(ROOT / "phase31_a0p8_summary.json"),
    ]
    points = _merge_points(summaries)
    print(f"Merged {len(points)} points: alphas = {[p['point'] for p in points]}")

    for attack in ("loss", "modent"):
        out = args.output_dir / f"phase31_breaking_point_fine_{attack}.png"
        _plot(points, attack, out, threshold=args.threshold)


if __name__ == "__main__":
    main()
