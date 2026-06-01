"""
Phase 2.4 orchestrator — cross-client contamination at round 20.

Conditions evaluated (rows of the final heatmap):
  baseline                  contributions/CIFAR10                            (round 20)
  retrain                   contributions/CIFAR10_RETRAINED_no0              (round 20)
  K05_cp / K10_cp / K15_cp  contributions/CIFAR10_WD_K<K>_class_pruning_p05_ft2  (round 20)
  K05_grad / K10_grad / K15_grad  contributions/CIFAR10_WD_K<K>_gradient            (round 20)

For each condition, scores c0's training samples (members) vs. test-split
non-members against c1/c2/c3's LOCAL models at round 20. Also fetches the
aggregate-level AUC at round 20 for the same condition as a side-by-side
comparison column.

Optional extras (enabled by default per user approval):
  - --baseline-per-round-trace : repeat per-client MIA at rounds {1,5,10,15,20}
                                  of CIFAR10/ to show when contamination develops
  - --include-ceiling          : score c0 against c0's OWN local model at round
                                  20 of CIFAR10/ as the direct-leakage ceiling
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PYTHON = sys.executable

DEFAULT_DATASET = "CIFAR10"
DEFAULT_MODEL = "resnet18"
DEFAULT_T_FINAL = 20
DEFAULT_TARGET_CLIENT = 0
DEFAULT_EVAL_CLIENTS = [1, 2, 3]
DEFAULT_TRACE_ROUNDS = [1, 5, 10, 15, 20]

# Condition table — order matters for the heatmap rows.
# (tag, db_name, source_db_for_member_ids_or_None)
CONDITIONS: List[Tuple[str, str, Optional[str]]] = [
    ("baseline",  "CIFAR10",                                    None),
    ("retrain",   "CIFAR10_RETRAINED_no0",                      "CIFAR10"),
    ("K05_cp",    "CIFAR10_WD_K05_class_pruning_p05_ft2",       "CIFAR10"),
    ("K10_cp",    "CIFAR10_WD_K10_class_pruning_p05_ft2",       "CIFAR10"),
    ("K15_cp",    "CIFAR10_WD_K15_class_pruning_p05_ft2",       "CIFAR10"),
    ("K05_grad",  "CIFAR10_WD_K05_gradient",                    "CIFAR10"),
    ("K10_grad",  "CIFAR10_WD_K10_gradient",                    "CIFAR10"),
    ("K15_grad",  "CIFAR10_WD_K15_gradient",                    "CIFAR10"),
]

# Map each condition's aggregate AUC source. JSON path + JSON-internal indexing.
# The "target" inside the existing per-round JSONs is either 'original' or 'unlearned'.
AGGREGATE_AUC_SOURCES: Dict[str, Tuple[str, str]] = {
    "baseline":   ("mia_original_cifar10.json",                          "original"),
    "retrain":    ("mia_retrain_cifar10.json",                           "original"),
    "K05_cp":     ("mia_phase23_K05_class_pruning_p05_ft2.json",         "original"),
    "K10_cp":     ("mia_phase23_K10_class_pruning_p05_ft2.json",         "original"),
    "K15_cp":     ("mia_phase23_K15_class_pruning_p05_ft2.json",         "original"),
    "K05_grad":   ("mia_phase23_K05_gradient.json",                      "original"),
    "K10_grad":   ("mia_phase23_K10_gradient.json",                      "original"),
    "K15_grad":   ("mia_phase23_K15_gradient.json",                      "original"),
}


def _run_per_client_mia(
    args,
    db_name: str,
    member_source_db: Optional[str],
    round_num: int,
    evaluate_clients: List[int],
    target_client: int,
    output_json: Path,
) -> Path:
    if output_json.exists() and not args.force:
        print(f"[phase24] {output_json} exists, reusing.")
        return output_json
    cmd = [
        PYTHON, "evaluate_mia_per_client.py",
        "--db-dir", "contributions",
        "--dataset", args.dataset,
        "--db-name", db_name,
        "--model", args.model,
        "--target-client", str(target_client),
        "--evaluate-clients", ",".join(str(c) for c in evaluate_clients),
        "--round", str(round_num),
        "--attacks", "loss,modent",
        "--n-members", "1000",
        "--n-nonmembers", "1000",
        "--non-member-source", "test",
        "--seed", str(args.seed),
        "--output", str(output_json),
    ]
    if member_source_db:
        cmd += ["--member-source-db-name", member_source_db]
    print(f"[phase24] {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"evaluate_mia_per_client.py failed (rc={rc}) for {db_name}")
    return output_json


def _read_auc(json_path: Path, client_id: int, attack: str) -> Optional[float]:
    data = json.loads(json_path.read_text())
    bucket = data.get("results", {}).get(str(client_id))
    if bucket is None:
        return None
    return float(bucket[attack]["members_vs_nonmembers"]["auc"])


def _read_aggregate_auc(json_path: Path, target: str, round_num: int, attack: str) -> Optional[float]:
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    rd = data.get("results", {}).get(str(round_num))
    if rd is None:
        return None
    return float(rd[target][attack]["members_vs_nonmembers"]["auc"])


def _collect_core(args) -> Dict[str, Dict]:
    """Returns {tag: {client_id (str): {attack: auc}, 'aggregate': auc}}."""
    out: Dict[str, Dict] = {}
    for tag, db_name, member_src in CONDITIONS:
        outfile = Path(f"mia_phase24_{tag}.json")
        _run_per_client_mia(
            args, db_name, member_src, args.t_final,
            args.evaluate_clients, args.target_client, outfile,
        )
        block: Dict = {}
        for cid in args.evaluate_clients:
            block[str(cid)] = {
                atk: _read_auc(outfile, cid, atk) for atk in ("loss", "modent")
            }
        # Aggregate AUC: read from the existing per-round JSONs if available.
        agg_src = AGGREGATE_AUC_SOURCES.get(tag)
        if agg_src is not None:
            jpath, jtarget = agg_src
            block["aggregate"] = _read_aggregate_auc(Path(jpath), jtarget, args.t_final, "loss")
        else:
            block["aggregate"] = None
        out[tag] = block
    return out


def _collect_ceiling(args) -> Optional[Dict]:
    """Score c0 against c0's OWN local model at round 20 of CIFAR10/. Upper-bound row."""
    outfile = Path("mia_phase24_ceiling_c0_local.json")
    if not outfile.exists() or args.force:
        cmd = [
            PYTHON, "evaluate_mia_per_client.py",
            "--db-dir", "contributions",
            "--dataset", args.dataset,
            "--db-name", "CIFAR10",
            "--model", args.model,
            "--target-client", str(args.target_client),
            "--evaluate-clients", str(args.target_client),  # SAME client = ceiling
            "--round", str(args.t_final),
            "--attacks", "loss,modent",
            "--n-members", "1000",
            "--n-nonmembers", "1000",
            "--non-member-source", "test",
            "--seed", str(args.seed),
            "--output", str(outfile),
        ]
        print(f"[phase24:ceiling] {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[phase24:ceiling] non-zero exit ({rc}); skipping ceiling row.")
            return None
    cid = args.target_client
    return {
        "loss": _read_auc(outfile, cid, "loss"),
        "modent": _read_auc(outfile, cid, "modent"),
    }


def _collect_baseline_trace(args) -> Optional[Dict[int, Dict]]:
    """{round: {client_id (str): {attack: auc}}} on the baseline DB across rounds."""
    out: Dict[int, Dict] = {}
    for r in args.trace_rounds:
        outfile = Path(f"mia_phase24_baseline_trace_round{r:02d}.json")
        _run_per_client_mia(
            args, "CIFAR10", None, r,
            args.evaluate_clients, args.target_client, outfile,
        )
        block: Dict = {}
        for cid in args.evaluate_clients:
            block[str(cid)] = {
                atk: _read_auc(outfile, cid, atk) for atk in ("loss", "modent")
            }
        out[r] = block
    return out


def _plot_heatmap(args, core: Dict, ceiling: Optional[Dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cols = [f"c{c}" for c in args.evaluate_clients] + ["mean", "aggregate"]
    row_order = [tag for tag, _, _ in CONDITIONS]
    if ceiling is not None:
        row_order = ["ceiling (c0 vs c0_local)"] + row_order

    data = np.full((len(row_order), len(cols)), float("nan"))
    annot = [["" for _ in cols] for _ in row_order]

    for i, tag in enumerate(row_order):
        if tag.startswith("ceiling"):
            v = ceiling.get("loss") if ceiling else None
            for j, _ in enumerate(args.evaluate_clients):
                pass  # leave NaN — ceiling only applies in its own row
            if v is not None:
                # Display the ceiling in the leftmost column ("c0_local self-leakage").
                data[i, 0] = v
                annot[i][0] = f"{v:.3f}"
                # Also put it in the 'mean' column as a single-value mean.
                data[i, len(args.evaluate_clients)] = v
                annot[i][len(args.evaluate_clients)] = f"{v:.3f}"
            continue
        block = core[tag]
        per_client_loss: List[float] = []
        for j, cid in enumerate(args.evaluate_clients):
            v = block[str(cid)]["loss"]
            data[i, j] = v
            annot[i][j] = f"{v:.3f}"
            per_client_loss.append(v)
        mean_v = float(np.mean(per_client_loss))
        data[i, len(args.evaluate_clients)] = mean_v
        annot[i][len(args.evaluate_clients)] = f"{mean_v:.3f}"
        agg = block.get("aggregate")
        if agg is not None:
            data[i, -1] = agg
            annot[i][-1] = f"{agg:.3f}"

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(row_order) + 1.5))
    finite = data[np.isfinite(data)]
    vmin = float(min(0.49, finite.min())) if finite.size else 0.49
    vmax = float(max(0.75, finite.max())) if finite.size else 0.75
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdYlGn_r",          # green=low (good), red=high (leaky)
        vmin=vmin, vmax=vmax,
    )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    for i in range(len(row_order)):
        for j in range(len(cols)):
            txt = annot[i][j]
            if txt:
                ax.text(j, i, txt, ha="center", va="center",
                        color=("white" if data[i, j] > (vmin + vmax) / 2 else "black"),
                        fontsize=9)
    fig.colorbar(im, ax=ax, label="MIA loss-attack AUC")
    ax.set_title(f"Phase 2.4 — Per-client local-model AUC vs c0's data @ round {args.t_final}")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"[phase24] Heatmap: {output}")


def _plot_baseline_trace(args, trace: Dict[int, Dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = sorted(trace.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    for cid in args.evaluate_clients:
        ys = [trace[r][str(cid)]["loss"] for r in rounds]
        ax.plot(rounds, ys, marker="o", linewidth=2.0,
                color=colors.get(cid, "tab:purple"),
                label=f"c{cid} local model")
    # Reference: aggregate AUC at the same rounds (from mia_original_cifar10.json)
    src = Path("mia_original_cifar10.json")
    if src.exists():
        agg = json.loads(src.read_text())
        ys = []
        for r in rounds:
            rd = agg.get("results", {}).get(str(r), {})
            mvn = rd.get("original", {}).get("loss", {}).get("members_vs_nonmembers")
            ys.append(mvn["auc"] if mvn else float("nan"))
        ax.plot(rounds, ys, "k--", linewidth=1.5, label="aggregate AUC (reference)")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0, label="random=0.5")
    ax.set_xlabel("Round")
    ax.set_ylabel("MIA loss-attack AUC")
    ax.set_title("Phase 2.4 — When does cross-client contamination develop? (baseline)")
    ax.set_xticks(rounds)
    ax.set_ylim(0.45, 0.80)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"[phase24] Trace plot: {output}")


def _print_summary(args, core: Dict, ceiling: Optional[Dict], trace: Optional[Dict]) -> None:
    print("\n[phase24] Loss-attack AUC summary @ round 20:")
    header = f"  {'condition':<24}" + "  ".join(f"c{c:>2}" for c in args.evaluate_clients) \
        + f"  {'mean':>6}  {'aggregate':>10}"
    print(header)
    for tag, _, _ in CONDITIONS:
        block = core[tag]
        per = [block[str(c)]["loss"] for c in args.evaluate_clients]
        mean = sum(per) / len(per)
        agg = block.get("aggregate")
        agg_str = f"{agg:.3f}" if agg is not None else "  n/a "
        cells = "  ".join(f"{v:>3.3f}" for v in per)
        print(f"  {tag:<24}{cells}  {mean:>6.3f}  {agg_str:>10}")
    if ceiling is not None and ceiling.get("loss") is not None:
        print(f"  {'ceiling (c0 vs c0_local)':<24}{ceiling['loss']:>3.3f}  (direct leakage upper bound)")
    if trace:
        print("\n[phase24] Baseline per-round trace (loss AUC):")
        for r in sorted(trace.keys()):
            per = [trace[r][str(c)]["loss"] for c in args.evaluate_clients]
            mean = sum(per) / len(per)
            cells = "  ".join(f"{v:>3.3f}" for v in per)
            print(f"  round {r:>2}  {cells}  mean={mean:.3f}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--t-final", type=int, default=DEFAULT_T_FINAL)
    p.add_argument("--target-client", type=int, default=DEFAULT_TARGET_CLIENT)
    p.add_argument("--evaluate-clients", type=str,
                   default=",".join(str(c) for c in DEFAULT_EVAL_CLIENTS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-per-round-trace", action="store_true", default=True,
                   help="Re-run per-client MIA at rounds {1,5,10,15,20} of CIFAR10.")
    p.add_argument("--trace-rounds", type=str,
                   default=",".join(str(r) for r in DEFAULT_TRACE_ROUNDS))
    p.add_argument("--include-ceiling", action="store_true", default=True,
                   help="Also score c0 against c0's own local model — direct-leakage ceiling.")
    p.add_argument("--output-heatmap", default="phase24_heatmap.png")
    p.add_argument("--output-trace", default="phase24_baseline_trace.png")
    p.add_argument("--output-summary", default="phase24_summary.json")
    p.add_argument("--force", action="store_true",
                   help="Re-run MIA even when per-condition JSONs exist.")
    args = p.parse_args()
    args.evaluate_clients = [int(x) for x in args.evaluate_clients.split(",") if x.strip()]
    args.trace_rounds = [int(x) for x in args.trace_rounds.split(",") if x.strip()]
    return args


def main():
    args = parse_args()
    print(f"[phase24] Conditions: {[t for t,_,_ in CONDITIONS]}")
    print(f"[phase24] Eval clients: {args.evaluate_clients}, target client: {args.target_client}")

    core = _collect_core(args)
    ceiling = _collect_ceiling(args) if args.include_ceiling else None
    trace = _collect_baseline_trace(args) if args.baseline_per_round_trace else None

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "t_final": args.t_final,
        "target_client": args.target_client,
        "evaluate_clients": args.evaluate_clients,
        "conditions": [tag for tag, _, _ in CONDITIONS],
        "core": core,
        "ceiling_c0_local": ceiling,
        "baseline_per_round_trace": trace,
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2, default=str))
    print(f"[phase24] Summary: {args.output_summary}")

    _print_summary(args, core, ceiling, trace)
    _plot_heatmap(args, core, ceiling, Path(args.output_heatmap))
    if trace:
        _plot_baseline_trace(args, trace, Path(args.output_trace))


if __name__ == "__main__":
    main()
