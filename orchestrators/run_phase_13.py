"""
Phase 1.3 driver — extended-horizon resurfacing to T=40.

Three continuation training runs, each spawning server.py + 3 clients (1/2/3,
excluding c0):
  1. Warm-start from CIFAR10_UNLEARNED_class_pruning_p05_ft2 round 5
     -> continue 35 rounds  -> CIFAR10_XHORIZON_K05_class_pruning_p05_ft2/ rounds 5..40
  2. Warm-start from CIFAR10_UNLEARNED_gradient round 5
     -> continue 35 rounds  -> CIFAR10_XHORIZON_K05_gradient/ rounds 5..40
  3. Warm-start from CIFAR10_RETRAINED_no0 round 20
     -> continue 20 rounds  -> CIFAR10_RETRAIN_XHORIZON/ rounds 20..40

Then per-round MIA on each of the 3 new DBs. Final plot overlays the 3 curves
with the retrain floor and original-FL ceiling as references.

Reuses the spawn helpers from retrain_baseline.py / run_phase_23.py.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from retrain_baseline import (
    PYTHON,
    SERVER_HOST,
    SERVER_PORT,
    _close_log,
    _kill,
    _spawn,
    _wait_for_server_port,
)

DEFAULT_DATASET = "CIFAR10"
DEFAULT_MODEL = "resnet18"
DEFAULT_TARGET_ROUND = 40  # T_total
DEFAULT_K = 5  # warm-start source round for cp / gradient (IID/horizontal only)
DEFAULT_NONIID_WARMSTART_ROUND = 20  # non-IID warm-starts at the unlearning event

# IID reference curves. Under --partition-mode=dirichlet these are skipped unless
# explicitly overridden (the in-experiment retrain_xhorizon curve is the reference).
REFERENCE_RETRAIN_JSON = Path("mia_retrain_cifar10.json")
REFERENCE_ORIGINAL_JSON = Path("mia_original_cifar10.json")


def _alpha_tag(alpha: float) -> str:
    """Match server.py's Dirichlet DB-naming: 0.5 -> 'a0p5', 1 -> 'a1'."""
    return f"{alpha:g}".replace(".", "p")


def _run_prefix(args) -> str:
    """Filename/identifier prefix so non-IID runs don't clobber IID artefacts."""
    if args.partition_mode == "dirichlet":
        return f"noniid_a{_alpha_tag(args.dirichlet_alpha)}_"
    return ""


@dataclass
class ContRun:
    tag: str              # e.g. "xhorizon_cp"
    source_db: str        # e.g. "CIFAR10_UNLEARNED_class_pruning_p05_ft2"
    source_round: int     # warm-start round
    out_suffix: str       # appended via --retrain-suffix
    out_db_name: str      # full contributions/<name>/
    n_rounds: int         # --num-rounds


def _build_runs(args) -> List[ContRun]:
    if args.partition_mode == "dirichlet":
        if args.dirichlet_alpha is None:
            raise SystemExit("[phase13] --dirichlet-alpha is required when --partition-mode=dirichlet")
        # Non-IID: all three curves warm-start at the round-20 unlearning event and
        # continue to T. The remaining clients still carry c0-correlated parameters
        # (Phase 3.4), so this is the configuration that can re-inject the signal.
        base = f"{args.dataset}_DIRICHLET_a{_alpha_tag(args.dirichlet_alpha)}"
        ws = args.noniid_warmstart_round
        return [
            ContRun(
                tag="xhorizon_cp",
                source_db=f"{base}_UNLEARNED_class_pruning_p05_ft2",
                source_round=ws,
                out_suffix="XHORIZON_class_pruning_p05_ft2",
                out_db_name=f"{base}_XHORIZON_class_pruning_p05_ft2",
                n_rounds=args.target_round - ws,
            ),
            ContRun(
                tag="xhorizon_gradient",
                source_db=f"{base}_UNLEARNED_gradient",
                source_round=ws,
                out_suffix="XHORIZON_gradient",
                out_db_name=f"{base}_XHORIZON_gradient",
                n_rounds=args.target_round - ws,
            ),
            ContRun(
                tag="retrain_xhorizon",
                source_db=f"{base}_RETRAINED_no0",
                source_round=20,
                out_suffix="RETRAIN_XHORIZON",
                out_db_name=f"{base}_RETRAIN_XHORIZON",
                n_rounds=args.target_round - 20,
            ),
        ]

    # Horizontal IID (original behaviour): cp/gradient warm-start at the K-sweep
    # withdrawal round, retrain at round 20.
    return [
        ContRun(
            tag="xhorizon_cp",
            source_db=f"{args.dataset}_UNLEARNED_class_pruning_p05_ft2",
            source_round=args.k,
            out_suffix=f"XHORIZON_K{args.k:02d}_class_pruning_p05_ft2",
            out_db_name=f"{args.dataset}_XHORIZON_K{args.k:02d}_class_pruning_p05_ft2",
            n_rounds=args.target_round - args.k,
        ),
        ContRun(
            tag="xhorizon_gradient",
            source_db=f"{args.dataset}_UNLEARNED_gradient",
            source_round=args.k,
            out_suffix=f"XHORIZON_K{args.k:02d}_gradient",
            out_db_name=f"{args.dataset}_XHORIZON_K{args.k:02d}_gradient",
            n_rounds=args.target_round - args.k,
        ),
        ContRun(
            tag="retrain_xhorizon",
            source_db=f"{args.dataset}_RETRAINED_no0",
            source_round=20,
            out_suffix="RETRAIN_XHORIZON",
            out_db_name=f"{args.dataset}_RETRAIN_XHORIZON",
            n_rounds=args.target_round - 20,
        ),
    ]


def _verify_source(run: ContRun) -> None:
    src_dir = Path("contributions") / run.source_db / f"round_{run.source_round:04d}"
    if not src_dir.is_dir():
        raise SystemExit(
            f"[phase13] Missing source: {src_dir}. Make sure Phase 2.2 unlearning / "
            f"retrain_baseline.py have produced an aggregate at round {run.source_round}."
        )
    aggs = list(src_dir.glob("*_aggregated_*_params.pkl"))
    if not aggs:
        raise SystemExit(f"[phase13] No aggregate pickles in {src_dir}")
    newest = max(aggs, key=lambda p: p.stat().st_mtime)
    print(f"[phase13] {run.tag}: source = {newest.name}")


def _continuation_done(run: ContRun, target_round: int) -> bool:
    """True iff the target final-round aggregate exists in run.out_db_name."""
    final = Path("contributions") / run.out_db_name / f"round_{target_round:04d}"
    return final.is_dir() and any(final.glob("*_aggregated_*_params.pkl"))


def _run_continuation(args, run: ContRun) -> None:
    """Spawn server.py + 3 clients for one extended continuation."""
    if _continuation_done(run, args.target_round) and not args.force_train:
        print(f"[phase13] {run.tag}: target round {args.target_round} aggregate present — skipping training.")
        return

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.partition_mode == "dirichlet":
        partition_flags = ["--partition-type", "dirichlet",
                           "--dirichlet-alpha", str(args.dirichlet_alpha)]
    else:
        partition_flags = ["--partition-type", "horizontal"]
    server_cmd = [
        PYTHON, "server.py",
        "--dataset", args.dataset,
        "--model", args.model,
        "--num-rounds", str(run.n_rounds),
        *partition_flags,
        "--retrain-suffix", run.out_suffix,
        "--init-weights-from-dataset", run.source_db,
        "--init-weights-from-round", str(run.source_round),
        "--first-round-offset", str(run.source_round),
        "--exclude-clients", str(args.exclude_client),
    ]
    print(f"[phase13] {run.tag}: server cmd = {' '.join(server_cmd)}")
    server_log = log_dir / f"server_{run.out_suffix}.log"
    server_proc = _spawn(server_cmd, server_log)

    client_procs = []
    try:
        _wait_for_server_port(SERVER_HOST, SERVER_PORT, timeout=args.server_start_timeout)
        print(f"[phase13] {run.tag}: server listening on {SERVER_HOST}:{SERVER_PORT}")
        client_ids = [i for i in range(args.num_clients) if i != args.exclude_client]
        for cid in client_ids:
            client_cmd = [
                PYTHON, "client.py",
                "--client-id", str(cid),
                "--num-clients", str(args.num_clients),
                "--dataset", args.dataset,
                "--model", args.model,
                *partition_flags,
                "--training-trace", "loss",
            ]
            if args.partition_seed is not None:
                client_cmd += ["--partition-seed", str(args.partition_seed)]
            client_log = log_dir / f"client_{cid}_{run.out_suffix}.log"
            client_procs.append(_spawn(client_cmd, client_log))
        rc = server_proc.wait()
        _close_log(server_proc)
        for p in client_procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _kill(p)
            _close_log(p)
        client_rcs = [p.returncode for p in client_procs]
        if rc != 0 or any(c != 0 for c in client_rcs):
            print(f"[phase13] {run.tag}: WARNING server rc={rc} clients={client_rcs}")
        else:
            print(f"[phase13] {run.tag}: training complete.")
    except BaseException:
        _kill(server_proc)
        for p in client_procs:
            _kill(p)
        raise


def _run_mia(args, run: ContRun) -> Path:
    """Per-round MIA over run.out_db_name. Member IDs come from the matching
    partition's baseline DB: the canonical CIFAR10/ under IID, but the
    CIFAR10_DIRICHLET_a<tag>/ baseline under non-IID (c0's partition is
    alpha-specific, so members must be drawn from the same Dirichlet draw)."""
    out_json = Path(f"mia_phase13_{_run_prefix(args)}{run.tag}.json")
    if out_json.exists() and not args.force:
        print(f"[phase13] {out_json} exists — reusing.")
        return out_json
    if args.partition_mode == "dirichlet":
        member_source_db = f"{args.dataset}_DIRICHLET_a{_alpha_tag(args.dirichlet_alpha)}"
    else:
        member_source_db = args.dataset
    cmd = [
        PYTHON, "evaluate_mia_per_round.py",
        "--db-dir", "contributions",
        "--dataset", args.dataset,
        "--db-name", run.out_db_name,
        "--member-source-db-name", member_source_db,
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
    print(f"[phase13] {run.tag}: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"[phase13] MIA failed for {run.tag} (rc={rc})")
    return out_json


def _read_curve(json_path: Path, attack: str = "loss",
                target: str = "original") -> Dict[int, float]:
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text())
    out: Dict[int, float] = {}
    for r_str, rd in (data.get("results") or {}).items():
        mvn = (rd.get(target) or {}).get(attack, {}).get("members_vs_nonmembers")
        if mvn and "auc" in mvn:
            out[int(r_str)] = float(mvn["auc"])
    return out


def _plot_horizon(args, curves: Dict[str, Dict[int, float]],
                  retrain_curve: Dict[int, float], original_curve: Dict[int, float],
                  attack: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    style = {
        "xhorizon_cp":     dict(color="tab:blue",   marker="o", label="class_pruning_p05_ft2 (K=5 + cont.)"),
        "xhorizon_gradient": dict(color="tab:orange", marker="s", label="gradient (K=5 + cont.)"),
        "retrain_xhorizon": dict(color="tab:green",  marker="^", label="retrain (no c0) + cont. from r=20"),
    }
    for tag, curve in curves.items():
        rs = sorted(curve.keys())
        ys = [curve[r] for r in rs]
        s = style.get(tag, dict(label=tag))
        ax.plot(rs, ys, linewidth=2.0, **s)

    # Reference horizontals at round 20 of the originals
    ceiling = original_curve.get(20)
    retrain_floor_at_20 = retrain_curve.get(20)
    if ceiling is not None:
        ax.axhline(ceiling, color="tab:red", linestyle="--",
                   linewidth=1.2, label=f"original-FL ceiling @ r20 ({ceiling:.3f})")
    if retrain_floor_at_20 is not None:
        ax.axhline(retrain_floor_at_20, color="tab:gray", linestyle="--",
                   linewidth=1.2, label=f"retrain floor @ r20 ({retrain_floor_at_20:.3f})")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, label="random=0.5")

    ax.set_xlabel("Round")
    ax.set_ylabel(f"MIA AUC ({attack})")
    ax.set_title(f"Phase 1.3 — Extended-horizon resurfacing (T={args.target_round}, K={args.k}) — {attack}")
    ax.set_ylim(0.45, 0.80)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[phase13] Plot: {output}")


def _summarise(args, curves: Dict[str, Dict[int, float]],
               retrain_curve: Dict[int, float], original_curve: Dict[int, float],
               attack: str) -> Dict:
    summary: Dict[str, dict] = {}
    for tag, curve in curves.items():
        rs = sorted(curve.keys())
        ys = [curve[r] for r in rs]
        summary[tag] = {
            "rounds": rs, "auc": ys,
            "first_auc": ys[0] if ys else None,
            "endpoint_auc": ys[-1] if ys else None,
            "min_auc": min(ys) if ys else None,
            "max_auc": max(ys) if ys else None,
        }
    summary["retrain_curve_round20"] = retrain_curve.get(20)
    summary["original_ceiling_round20"] = original_curve.get(20)
    summary["attack"] = attack
    return summary


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--num-clients", type=int, default=4)
    p.add_argument("--target-round", type=int, default=DEFAULT_TARGET_ROUND,
                   help="Total final round T (default 40).")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help="Warm-start round K for cp/gradient under IID (default 5). "
                        "Ignored under --partition-mode=dirichlet.")
    p.add_argument("--partition-mode", choices=["horizontal", "dirichlet"],
                   default="horizontal",
                   help="horizontal=IID (original Phase 1.3); dirichlet=non-IID "
                        "resurfacing test where remaining clients still carry "
                        "c0-correlated parameters.")
    p.add_argument("--dirichlet-alpha", type=float, default=None,
                   help="Dirichlet concentration; required when --partition-mode=dirichlet.")
    p.add_argument("--noniid-warmstart-round", type=int,
                   default=DEFAULT_NONIID_WARMSTART_ROUND,
                   help="Warm-start round for all curves under dirichlet (default 20, "
                        "the unlearning event).")
    p.add_argument("--exclude-client", type=int, default=0)
    p.add_argument("--partition-seed", type=int, default=None)
    p.add_argument("--reference-retrain-json", default=None,
                   help="Override the retrain reference-curve JSON. Defaults to the "
                        "IID file under horizontal mode, skipped under dirichlet.")
    p.add_argument("--reference-original-json", default=None,
                   help="Override the original-FL reference-curve JSON. Defaults to the "
                        "IID file under horizontal mode, skipped under dirichlet.")
    p.add_argument("--log-dir", default="logs/phase13")
    p.add_argument("--server-start-timeout", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-plot", default=None)
    p.add_argument("--output-summary", default=None)
    p.add_argument("--force", action="store_true",
                   help="Re-score MIA even if JSONs exist.")
    p.add_argument("--force-train", action="store_true",
                   help="Re-train even if target-round aggregate exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate sources and exit before spawning anything.")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training, only run MIA + plot on existing DBs.")
    return p.parse_args()


def main():
    args = parse_args()

    # Mode-aware output names so non-IID runs don't clobber IID artefacts.
    prefix = _run_prefix(args)
    if args.output_plot is None:
        args.output_plot = f"phase13_{prefix}extended_horizon.png"
    if args.output_summary is None:
        args.output_summary = f"phase13_{prefix}summary.json"

    # Reference curves: IID files by default under horizontal; skipped under
    # dirichlet (the in-experiment retrain_xhorizon curve is the reference)
    # unless the caller points at non-IID baseline MIA JSONs.
    if args.reference_retrain_json is not None:
        retrain_ref = Path(args.reference_retrain_json)
    elif args.partition_mode == "horizontal":
        retrain_ref = REFERENCE_RETRAIN_JSON
    else:
        retrain_ref = Path(f"__no_reference_{prefix}retrain__")
    if args.reference_original_json is not None:
        original_ref = Path(args.reference_original_json)
    elif args.partition_mode == "horizontal":
        original_ref = REFERENCE_ORIGINAL_JSON
    else:
        original_ref = Path(f"__no_reference_{prefix}original__")

    runs = _build_runs(args)
    for r in runs:
        _verify_source(r)
    if args.dry_run:
        print("[phase13] --dry-run: sources OK, exiting.")
        return

    # 1. Continuation training (3 runs, sequential to avoid port collision on 8080).
    if not args.skip_train:
        for run in runs:
            t0 = time.monotonic()
            _run_continuation(args, run)
            print(f"[phase13] {run.tag}: training elapsed {(time.monotonic()-t0)/60:.1f} min")
    else:
        print("[phase13] --skip-train: jumping to MIA pass.")

    # 2. MIA on each
    mia_paths = {}
    for run in runs:
        mia_paths[run.tag] = _run_mia(args, run)

    # 3. Build curves + plot
    curves_loss = {tag: _read_curve(p, "loss") for tag, p in mia_paths.items()}
    curves_modent = {tag: _read_curve(p, "modent") for tag, p in mia_paths.items()}

    retrain_loss = _read_curve(retrain_ref, "loss")
    original_loss = _read_curve(original_ref, "loss")
    retrain_modent = _read_curve(retrain_ref, "modent")
    original_modent = _read_curve(original_ref, "modent")

    _plot_horizon(args, curves_loss, retrain_loss, original_loss, "loss",
                  Path(args.output_plot))
    _plot_horizon(args, curves_modent, retrain_modent, original_modent, "modent",
                  Path(str(args.output_plot).replace(".png", "_modent.png")))

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "k": args.k,
        "target_round": args.target_round,
        "loss": _summarise(args, curves_loss, retrain_loss, original_loss, "loss"),
        "modent": _summarise(args, curves_modent, retrain_modent, original_modent, "modent"),
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2))
    print(f"[phase13] Summary: {args.output_summary}")

    # Decision-rule auto-readout
    cp_endpoint = (curves_loss.get("xhorizon_cp") or {}).get(args.target_round, float("nan"))
    gr_endpoint = (curves_loss.get("xhorizon_gradient") or {}).get(args.target_round, float("nan"))
    rt_endpoint = (curves_loss.get("retrain_xhorizon") or {}).get(args.target_round, float("nan"))
    print(f"\n[phase13] Endpoint AUC @ round {args.target_round} (loss):")
    print(f"  cp:               {cp_endpoint:.3f}")
    print(f"  gradient:         {gr_endpoint:.3f}")
    print(f"  retrain-extended: {rt_endpoint:.3f}")
    print(f"  delta(cp - retrain):       {cp_endpoint - rt_endpoint:+.3f}")
    print(f"  delta(gradient - retrain): {gr_endpoint - rt_endpoint:+.3f}")


if __name__ == "__main__":
    main()
