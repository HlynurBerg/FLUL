"""
Phase 2.3 MIA evaluation + final plot.

For each (K, algo) in the Phase 2.3 sweep:
  - K in {5, 10, 15}: continuation DB CIFAR10_WD_K<K>_<algo> — MIA at round 20.
  - K = 20: already in Phase 2.2's mia_unlearned_<algo>.json — read directly.

Also pulls:
  - Retrain baseline @ round 20 from mia_retrain_cifar10.json (horizontal line).
  - Original FL @ round 20 from the same JSON's "original" branch (reference).

Output:
  - mia_phase23_<algo>.json       per-algorithm MIA results indexed by K
  - phase23_withdraw_timing.png   AUC vs K plot, two curves + reference lines
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PYTHON = sys.executable

DEFAULT_DATASET = "CIFAR10"
DEFAULT_MODEL = "resnet18"
DEFAULT_K_LIST = [5, 10, 15]
DEFAULT_ALGOS = ["class_pruning_p05_ft2", "gradient"]
DEFAULT_K_FINAL = 20  # round T at which we evaluate


def _continuation_db_name(dataset: str, k: int, algo: str) -> str:
    return f"{dataset}_WD_K{k:02d}_{algo}"


def _existing_unlearned_json(algo: str) -> Path:
    """Phase 2.2 result JSONs for the K=T (no withdrawal) case."""
    return Path(f"mia_unlearned_{algo}.json")


def _run_mia_on_continuation(args, k: int, algo: str) -> Path:
    """Run evaluate_mia_per_round.py at round T against the continuation DB."""
    db_name = _continuation_db_name(args.dataset, k, algo)
    out_json = Path(f"mia_phase23_K{k:02d}_{algo}.json")
    if out_json.exists() and not args.force:
        print(f"[phase23-mia] K={k} algo={algo}: reusing {out_json}")
        return out_json

    cmd = [
        PYTHON, "evaluate_mia_per_round.py",
        "--db-dir", "contributions",
        "--dataset", args.dataset,
        "--db-name", db_name,
        "--member-source-db-name", args.dataset,   # c0 only has manifests in the original DB
        "--model", args.model,
        "--target-client", str(args.target_client),
        "--target", "original",                     # continuation DB has only one aggregate per round
        "--attacks", "loss,modent",
        "--n-members", "1000",
        "--n-nonmembers", "1000",
        "--non-member-source", "test",
        "--seed", str(args.seed),
        "--rounds", str(args.t_final),
        "--output", str(out_json),
    ]
    print(f"[phase23-mia] K={k} algo={algo}: cmd = {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"evaluate_mia_per_round.py failed (rc={rc}) for K={k} algo={algo}")
    return out_json


def _auc_at(json_path: Path, round_t: int, target: str, attack: str) -> float:
    """Extract one AUC from a per-round MIA JSON."""
    data = json.loads(json_path.read_text())
    rd = data["results"].get(str(round_t))
    if rd is None:
        raise KeyError(f"Round {round_t} missing in {json_path}")
    return float(rd[target][attack]["members_vs_nonmembers"]["auc"])


def _collect_curves(args) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return {algo: {K: {attack: auc}}} for both algos and all K (5,10,15,20)."""
    curves: Dict[str, Dict[int, Dict[str, float]]] = {a: {} for a in args.algos}
    for algo in args.algos:
        # K < T: continuation DB
        for k in args.k_list:
            j = _run_mia_on_continuation(args, k, algo)
            curves[algo][k] = {
                atk: _auc_at(j, args.t_final, "original", atk)
                for atk in ("loss", "modent")
            }
        # K == T: Phase 2.2 result (target = 'unlearned' branch in the existing JSON)
        existing = _existing_unlearned_json(algo)
        if not existing.exists():
            raise SystemExit(
                f"Missing Phase 2.2 JSON {existing} for K={args.t_final} (algo={algo}). "
                "Required to anchor the right-hand side of the plot."
            )
        curves[algo][args.t_final] = {
            atk: _auc_at(existing, args.t_final, "unlearned", atk)
            for atk in ("loss", "modent")
        }
    return curves


def _reference_lines(args) -> Dict[str, float]:
    """Return {label: AUC@T} for reference horizontal lines."""
    refs: Dict[str, float] = {}
    retrain = Path("mia_retrain_cifar10.json")
    if retrain.exists():
        refs["retrain (no c0 ever)"] = _auc_at(retrain, args.t_final, "original", "loss")
    orig = Path("mia_original_cifar10.json")
    if orig.exists():
        refs["original FL (c0 in 1..T, no unlearn)"] = _auc_at(orig, args.t_final, "original", "loss")
    return refs


def _plot(args, curves, refs, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    all_ks = sorted(set().union(*[set(curves[a].keys()) for a in curves]))
    colors = {"class_pruning_p05_ft2": "tab:green", "gradient": "tab:orange"}
    markers = {"class_pruning_p05_ft2": "o", "gradient": "s"}
    pretty = {
        "class_pruning_p05_ft2": "class_pruning (p=0.05, ft=2)",
        "gradient": "gradient",
    }
    for algo in args.algos:
        ys = [curves[algo][k]["loss"] for k in all_ks]
        ax.plot(
            all_ks, ys,
            color=colors.get(algo, "tab:blue"),
            marker=markers.get(algo, "x"),
            linewidth=2.0, markersize=8,
            label=f"unlearn(K) + continue: {pretty.get(algo, algo)}",
        )

    ref_styles = [(":", "tab:gray"), ("--", "tab:red")]
    for (label, val), (ls, c) in zip(refs.items(), ref_styles):
        ax.axhline(val, linestyle=ls, color=c, linewidth=1.5, label=f"{label} @ T={args.t_final}: AUC={val:.3f}")

    ax.set_xlabel("K (round c0 withdrew & got unlearned)")
    ax.set_ylabel(f"MIA loss-attack AUC at round T={args.t_final}")
    ax.set_title("Phase 2.3 — Withdrawal Timing (CIFAR-10 ResNet-18, 4 clients)")
    ax.set_xticks(all_ks)
    ax.set_ylim(0.45, max(0.78, max(refs.values()) + 0.02 if refs else 0.75))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"[phase23-mia] Plot saved: {output}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--k-list", type=str, default=",".join(str(x) for x in DEFAULT_K_LIST))
    p.add_argument("--algos", type=str, default=",".join(DEFAULT_ALGOS))
    p.add_argument("--t-final", type=int, default=DEFAULT_K_FINAL)
    p.add_argument("--target-client", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-plot", default="phase23_withdraw_timing.png")
    p.add_argument("--output-summary", default="phase23_summary.json")
    p.add_argument("--force", action="store_true", help="Re-run MIA even if output JSONs exist.")
    args = p.parse_args()
    args.k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    args.algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    return args


def main():
    args = parse_args()
    curves = _collect_curves(args)
    refs = _reference_lines(args)

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "t_final": args.t_final,
        "algos": args.algos,
        "k_list": sorted(set(args.k_list + [args.t_final])),
        "curves": curves,
        "references": refs,
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2))
    print(f"[phase23-mia] Summary: {args.output_summary}")

    print("\n[phase23-mia] AUC @ T={} (loss attack):".format(args.t_final))
    print(f"  {'K':>4}  ", "  ".join(f"{a:>26}" for a in args.algos))
    for k in sorted(set(args.k_list + [args.t_final])):
        cells = [f"{curves[a][k]['loss']:.4f}" for a in args.algos]
        print(f"  {k:>4}  ", "  ".join(f"{c:>26}" for c in cells))
    for label, val in refs.items():
        print(f"  ref: {label}: {val:.4f}")

    _plot(args, curves, refs, Path(args.output_plot))


if __name__ == "__main__":
    main()
