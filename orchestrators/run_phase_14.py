"""
Phase 1.4 driver — per-class breakdown of local-model leakage under non-IID
(α=0.1, CIFAR-10 ResNet-18). Tests the "data-overlap drives contamination"
hypothesis behind Phase 3.4's c3 outlier (0.652 vs c1/c2's 0.368/0.409 against c0).

Steps:
  1. Reproduce the per-client Dirichlet(α=0.1) partition (seed=42) and tabulate
     per-client class histograms — written to phase14_class_histograms.json.
  2. For each (target_class k, evaluated client c) in {0..9} x {1,2,3}:
       evaluate_mia_per_client.py --db-name CIFAR10_DIRICHLET_a0p1
                                   --member-source-db-name CIFAR10_DIRICHLET_a0p1
                                   --target-client 0 --evaluate-clients <c>
                                   --round 20 --member-class-filter k
                                   --output mia_phase14_c<c>_class<k>.json
  3. Aggregate the 3x10 AUC matrix (loss attack) and plot:
       - phase14_per_class_heatmap.png  : 3 (clients) x 10 (c0-classes)
       - phase14_auc_vs_overlap.png     : per-client AUC-vs-class-overlap scatter
  4. Write phase14_summary.json with the full table + correlation coefficients.
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
from typing import Dict, List, Optional

PYTHON = sys.executable

DEFAULT_DATASET = "CIFAR10"
DEFAULT_DB_NAME = "CIFAR10_DIRICHLET_a0p1"
DEFAULT_ALPHA = 0.1
DEFAULT_PARTITION_SEED = 42
DEFAULT_NUM_CLIENTS = 4
DEFAULT_NUM_CLASSES = 10
DEFAULT_TARGET_CLIENT = 0
DEFAULT_EVALUATE_CLIENTS = [1, 2, 3]
DEFAULT_ROUND = 20
DEFAULT_MODEL = "resnet18"


def _compute_class_histograms(dataset: str, alpha: float, seed: int,
                              num_clients: int, num_classes: int) -> Dict[int, Dict[int, int]]:
    """Re-run the per-class Dirichlet partition and tabulate per-client class counts.

    Returns {client_id: {class_id: count}} matching the deterministic partition
    used by client.py at α and seed.
    """
    from torchvision import datasets, transforms
    from utils import _dirichlet_partition_indices, _extract_targets

    if dataset == "CIFAR10":
        ds = datasets.CIFAR10(root="./data", train=True, download=False,
                              transform=transforms.ToTensor())
    elif dataset == "MNIST":
        ds = datasets.MNIST(root="./data", train=True, download=False,
                            transform=transforms.ToTensor())
    elif dataset == "FASHIONMNIST":
        ds = datasets.FashionMNIST(root="./data", train=True, download=False,
                                   transform=transforms.ToTensor())
    else:
        raise SystemExit(f"Unsupported dataset for class histograms: {dataset}")

    targets = _extract_targets(ds)
    indices_per_client = _dirichlet_partition_indices(ds, num_clients, alpha, seed)
    histograms: Dict[int, Dict[int, int]] = {}
    for cid, ix in enumerate(indices_per_client):
        h = {k: 0 for k in range(num_classes)}
        for i in ix:
            h[int(targets[i])] += 1
        histograms[cid] = h
    return histograms


def _run_mia_one(args, evaluate_client: int, class_filter: int) -> Path:
    out_json = Path(f"mia_phase14_c{evaluate_client}_class{class_filter}.json")
    if out_json.exists() and not args.force:
        print(f"[phase14] {out_json} exists — reusing.")
        return out_json
    cmd = [
        PYTHON, "evaluate_mia_per_client.py",
        "--db-dir", "contributions",
        "--dataset", args.dataset,
        "--db-name", args.db_name,
        "--member-source-db-name", args.db_name,
        "--model", args.model,
        "--target-client", str(args.target_client),
        "--evaluate-clients", str(evaluate_client),
        "--round", str(args.round),
        "--attacks", "loss,modent",
        "--n-members", str(args.n_members),
        "--n-nonmembers", str(args.n_nonmembers),
        "--non-member-source", "test",
        "--seed", str(args.seed),
        "--member-class-filter", str(class_filter),
        "--output", str(out_json),
    ]
    print(f"[phase14] c{evaluate_client} class{class_filter}: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        # Some classes may have too-few c0 samples under extreme skew. Don't crash;
        # mark as NaN and continue.
        print(f"[phase14] c{evaluate_client} class{class_filter} FAILED (rc={rc}) — skipping.")
        return out_json
    return out_json


def _read_auc(json_path: Path, client_id: int, attack: str) -> Optional[float]:
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return None
    bucket = (data.get("results") or {}).get(str(client_id))
    if bucket is None:
        return None
    mvn = bucket.get(attack, {}).get("members_vs_nonmembers")
    if mvn and "auc" in mvn:
        return float(mvn["auc"])
    return None


def _plot_heatmap(args, table: Dict[int, Dict[int, float]],
                  histograms: Dict[int, Dict[int, int]],
                  attack: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    classes = sorted(histograms[args.target_client].keys())
    eval_clients = args.evaluate_clients

    data = np.full((len(eval_clients), len(classes)), float("nan"))
    annot = [["" for _ in classes] for _ in eval_clients]
    for i, c in enumerate(eval_clients):
        for j, k in enumerate(classes):
            v = (table.get(c) or {}).get(k)
            if v is not None:
                data[i, j] = v
                annot[i][j] = f"{v:.2f}"

    fig, (ax_h, ax_top) = plt.subplots(
        nrows=2, ncols=1, figsize=(11, 5.5),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    # Heatmap
    finite = data[np.isfinite(data)]
    vmin = float(min(0.45, finite.min())) if finite.size else 0.45
    vmax = float(max(0.75, finite.max())) if finite.size else 0.75
    im = ax_h.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
    ax_h.set_xticks(range(len(classes)))
    ax_h.set_xticklabels([f"k={k}" for k in classes])
    ax_h.set_yticks(range(len(eval_clients)))
    ax_h.set_yticklabels([f"c{c}" for c in eval_clients])
    for i in range(len(eval_clients)):
        for j in range(len(classes)):
            txt = annot[i][j]
            if txt:
                ax_h.text(j, i, txt, ha="center", va="center",
                          color=("white" if data[i, j] > (vmin + vmax) / 2 else "black"),
                          fontsize=9)
    fig.colorbar(im, ax=ax_h, label=f"AUC ({attack})")
    ax_h.set_title(f"Phase 1.4 -- per-class local-model AUC at alpha={args.alpha}, round {args.round}")
    ax_h.set_ylabel("Evaluated client")

    # Top panel: per-class c0 sample count for context
    c0_hist = histograms[args.target_client]
    ax_top.bar(range(len(classes)), [c0_hist[k] for k in classes], color="tab:gray")
    ax_top.set_ylabel(f"c{args.target_client} count")
    ax_top.set_xlabel("c0-class")
    ax_top.set_xticks(range(len(classes)))
    ax_top.set_xticklabels([f"k={k}" for k in classes])
    ax_top.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[phase14] Heatmap: {output}")


def _plot_scatter(args, table: Dict[int, Dict[int, float]],
                  histograms: Dict[int, Dict[int, int]],
                  attack: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    correlations = {}

    for c in args.evaluate_clients:
        xs, ys = [], []
        hist = histograms[c]
        for k in sorted(hist.keys()):
            auc = (table.get(c) or {}).get(k)
            if auc is None:
                continue
            xs.append(hist[k])
            ys.append(auc)
        if not xs:
            continue
        ax.scatter(xs, ys, label=f"c{c}", color=colors.get(c, "tab:purple"),
                   s=70, alpha=0.85, edgecolor="black", linewidth=0.4)
        # Annotate each point with the class index
        hist_sorted = [(k, hist[k]) for k in sorted(hist.keys())
                       if (table.get(c) or {}).get(k) is not None]
        for (k, _), x, y in zip(hist_sorted, xs, ys):
            ax.annotate(f"k={k}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=colors.get(c, "tab:purple"))
        if len(xs) >= 2 and np.std(xs) > 0 and np.std(ys) > 0:
            r = float(np.corrcoef(xs, ys)[0, 1])
        else:
            r = float("nan")
        correlations[c] = r

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel(f"c0-class sample count in evaluated client (alpha={args.alpha})")
    ax.set_ylabel(f"MIA AUC ({attack})")
    title_corr = "  ".join(f"r(c{c})={correlations.get(c, float('nan')):.2f}"
                           for c in args.evaluate_clients)
    ax.set_title(f"Phase 1.4 — AUC vs class-overlap   [{title_corr}]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[phase14] Scatter: {output}")
    return correlations


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--db-name", default=DEFAULT_DB_NAME)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--partition-seed", type=int, default=DEFAULT_PARTITION_SEED)
    p.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    p.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    p.add_argument("--target-client", type=int, default=DEFAULT_TARGET_CLIENT)
    p.add_argument("--evaluate-clients", type=str,
                   default=",".join(str(c) for c in DEFAULT_EVALUATE_CLIENTS))
    p.add_argument("--round", type=int, default=DEFAULT_ROUND)
    p.add_argument("--n-members", type=int, default=300,
                   help="Per-class members. CIFAR-10 c0 typically has 50-500 per class at α=0.1.")
    p.add_argument("--n-nonmembers", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-heatmap", default="phase14_per_class_heatmap.png")
    p.add_argument("--output-scatter", default="phase14_auc_vs_overlap.png")
    p.add_argument("--output-histograms", default="phase14_class_histograms.json")
    p.add_argument("--output-summary", default="phase14_summary.json")
    p.add_argument("--force", action="store_true",
                   help="Re-score even if per-(c,k) JSONs exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute histograms and print planned MIA invocations; don't run them.")
    args = p.parse_args()
    args.evaluate_clients = [int(x) for x in args.evaluate_clients.split(",") if x.strip()]
    return args


def main():
    args = parse_args()
    print(f"[phase14] alpha={args.alpha} seed={args.partition_seed} db_name={args.db_name}")

    # 1. Per-client class histograms
    print("[phase14] Computing per-client class histograms ...")
    histograms = _compute_class_histograms(
        args.dataset, args.alpha, args.partition_seed,
        args.num_clients, args.num_classes,
    )
    Path(args.output_histograms).write_text(json.dumps(histograms, indent=2))
    print(f"[phase14] Histograms: {args.output_histograms}")
    for cid in range(args.num_clients):
        total = sum(histograms[cid].values())
        nonzero = sum(1 for v in histograms[cid].values() if v > 0)
        print(f"  c{cid}: total={total:>6}  classes_present={nonzero:>2}/{args.num_classes}  "
              + " ".join(f"k{k}:{histograms[cid][k]:>5}" for k in sorted(histograms[cid])))

    if args.dry_run:
        print("[phase14] --dry-run: histograms computed, exiting before MIA pass.")
        return

    # 2. MIA pass: 3 eval-clients x 10 classes = 30 calls
    t0 = time.monotonic()
    json_paths: Dict[int, Dict[int, Path]] = {c: {} for c in args.evaluate_clients}
    for c in args.evaluate_clients:
        for k in sorted(histograms[args.target_client].keys()):
            if histograms[args.target_client][k] == 0:
                print(f"[phase14] c0 has 0 samples of class {k}; skipping MIA for (c{c}, k={k}).")
                continue
            json_paths[c][k] = _run_mia_one(args, c, k)
    print(f"[phase14] MIA pass: {(time.monotonic()-t0)/60:.1f} min")

    # 3. Build the table
    table_loss: Dict[int, Dict[int, Optional[float]]] = {c: {} for c in args.evaluate_clients}
    table_modent: Dict[int, Dict[int, Optional[float]]] = {c: {} for c in args.evaluate_clients}
    for c, kdict in json_paths.items():
        for k, p in kdict.items():
            table_loss[c][k] = _read_auc(p, c, "loss")
            table_modent[c][k] = _read_auc(p, c, "modent")

    # 4. Plot
    corr_loss = _plot_scatter(args, table_loss, histograms, "loss",
                              Path(args.output_scatter))
    _plot_heatmap(args, table_loss, histograms, "loss", Path(args.output_heatmap))
    _plot_heatmap(args, table_modent, histograms, "modent",
                  Path(str(args.output_heatmap).replace(".png", "_modent.png")))

    # 5. Summary
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "alpha": args.alpha,
        "partition_seed": args.partition_seed,
        "db_name": args.db_name,
        "round": args.round,
        "target_client": args.target_client,
        "evaluate_clients": args.evaluate_clients,
        "histograms": {str(c): {str(k): v for k, v in d.items()}
                       for c, d in histograms.items()},
        "auc_table_loss": {str(c): {str(k): v for k, v in d.items()}
                            for c, d in table_loss.items()},
        "auc_table_modent": {str(c): {str(k): v for k, v in d.items()}
                              for c, d in table_modent.items()},
        "correlations_loss": {str(c): corr_loss.get(c) for c in args.evaluate_clients},
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2))
    print(f"[phase14] Summary: {args.output_summary}")

    # Headline table
    print("\n[phase14] AUC table (loss attack):")
    header = "  client  " + "  ".join(f"k={k:<3}" for k in sorted(histograms[args.target_client]))
    print(header)
    for c in args.evaluate_clients:
        cells = []
        for k in sorted(histograms[args.target_client]):
            v = table_loss.get(c, {}).get(k)
            cells.append(f"{v:>5.3f}" if v is not None else "  —  ")
        print(f"  c{c:<5}  " + "  ".join(cells))
    print("\n[phase14] Pearson r(AUC, c0-class-count-in-c):")
    for c, r in corr_loss.items():
        print(f"  c{c}: r={r:.3f}")


if __name__ == "__main__":
    main()
