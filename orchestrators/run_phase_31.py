"""
Phase 3.1 / 3.2 sweep orchestrator - heterogeneity breaking point.

For each heterogeneity value (Dirichlet alpha OR class_vertical primary/secondary
share), this script drives:

  1. Baseline FL training            -> contributions/<DB>/
  2. Retrain baseline (no c0) + MIA  -> contributions/<DB>_RETRAINED_no0/
                                        + mia_phase3X_<tag>_retrain.json
  3. Per-algorithm unlearning + MIA  -> contributions/<DB>_UNLEARNED_<algo>/
                                        + mia_unlearned_<algo>.json
                                        (via run_phase_d.py)

After all values are processed, it assembles a summary JSON keyed by
heterogeneity value x algorithm, containing post-unlearning MIA AUC (loss +
modent), test accuracy of the round-20 aggregate, and the gap vs. the retrain
baseline. ``--dry-run`` prints the planned commands and expected DB paths
without spawning anything.

Example (Phase 3.1):
    python run_phase_31.py --partition-mode dirichlet \\
        --dataset CIFAR10 --model resnet18 --num-clients 4 --num-rounds 20 \\
        --partition-seed 42 --target-client 0 \\
        --alpha-grid 1000,10,1.0,0.5,0.1,0.05 \\
        --algorithms gradient,influence,class_pruning_p05_ft2 \\
        --sanity-algos hessian,gradient_ascent_kd --sanity-alpha 1000 \\
        --results-dir results/phase31 --summary phase31_summary.json

Example (Phase 3.2):
    python run_phase_31.py --partition-mode class_vertical \\
        --dataset CUSTOM --dataset-path Information/banana \\
        --img-size 416 --num-channels 3 --num-classes 4 \\
        --model simplenet --num-clients 4 --num-rounds 20 \\
        --partition-seed 42 --target-client 0 \\
        --share-grid "0.25:0.25,0.40:0.20,0.60:0.13,0.85:0.05,0.95:0.017,1.0:0.0" \\
        --algorithms gradient,influence,class_pruning_p05_ft2 \\
        --results-dir results/phase32 --summary phase32_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from contributions_db import ContributionDB
from model import create_model, set_parameters
from utils import load_data, test

from retrain_baseline import (
    PYTHON,
    SERVER_HOST,
    SERVER_PORT,
    _close_log,
    _kill,
    _spawn,
    _wait_for_server_port,
)

DEFAULT_ALPHA_GRID = "1000,10,1.0,0.5,0.1,0.05"
DEFAULT_SHARE_GRID = "0.25:0.25,0.40:0.20,0.60:0.13,0.85:0.05,0.95:0.017,1.0:0.0"
DEFAULT_ALGOS = "gradient,influence,class_pruning_p05_ft2"


def _alpha_tag(alpha: float) -> str:
    """Mirrors server.py's alpha-tag (e.g. 0.1 -> '0p1', 1000 -> '1000')."""
    return f"{alpha:g}".replace(".", "p")


def _share_tag(primary: float, secondary: float) -> str:
    """Mirrors server.py's class_vertical p/s suffix."""
    p = f"{primary:.2f}".rstrip("0").rstrip(".")
    s = f"{secondary:.2f}".rstrip("0").rstrip(".")
    return f"p{p}_s{s}"


def _db_name_baseline(args, point) -> str:
    if args.partition_mode == "dirichlet":
        return f"{args.dataset}_DIRICHLET_a{_alpha_tag(point)}"
    primary, secondary = point
    # Match server.py's back-compat for the default 85/5 split.
    if abs(primary - 0.85) < 1e-9 and abs(secondary - 0.05) < 1e-9:
        return f"{args.dataset}_CLASS_VERTICAL"
    return f"{args.dataset}_CLASS_VERTICAL_{_share_tag(primary, secondary)}"


def _db_name_retrain(args, point) -> str:
    return f"{_db_name_baseline(args, point)}_RETRAINED_no{args.target_client}"


def _db_name_unlearned(args, point, algo: str) -> str:
    return f"{_db_name_baseline(args, point)}_UNLEARNED_{algo}"


def _partition_flags(args, point) -> List[str]:
    """Partition flags safe for BOTH server.py and client.py. Note that
    --partition-seed is intentionally NOT included here - server.py does not
    accept it (it only affects per-client partitioning), so callers must add
    it to client commands only."""
    flags = []
    if args.partition_mode == "dirichlet":
        flags += [
            "--partition-type", "dirichlet",
            "--dirichlet-alpha", str(point),
        ]
    else:
        primary, secondary = point
        flags += [
            "--partition-type", "class_vertical",
            "--primary-share", str(primary),
            "--secondary-share", str(secondary),
        ]
    return flags


def _partition_seed_flags(args) -> List[str]:
    """Client-only partition-seed flag."""
    if args.partition_seed is None:
        return []
    return ["--partition-seed", str(args.partition_seed)]


def _dataset_flags(args) -> List[str]:
    flags = ["--dataset", args.dataset, "--model", args.model]
    if args.dataset_path:
        flags += ["--dataset-path", args.dataset_path]
    if args.img_size is not None:
        flags += ["--img-size", str(args.img_size)]
    if args.num_channels is not None:
        flags += ["--num-channels", str(args.num_channels)]
    if args.num_classes is not None:
        flags += ["--num-classes", str(args.num_classes)]
    return flags


def _point_tag(args, point) -> str:
    if args.partition_mode == "dirichlet":
        return f"a{_alpha_tag(point)}"
    primary, secondary = point
    return _share_tag(primary, secondary)


# -----------------------------------------------------------------------------
# Training / retrain spawners
# -----------------------------------------------------------------------------

def _spawn_baseline_training(args, point, log_dir: Path) -> int:
    """Spawn run_training.py-style server + N clients. Return server rc."""
    server_cmd = [
        PYTHON, "server.py",
        *_dataset_flags(args),
        *_partition_flags(args, point),
        "--num-rounds", str(args.num_rounds),
    ]
    tag = _point_tag(args, point)
    server_log = log_dir / f"server_baseline_{tag}.log"
    print(f"[phase3] Baseline: {' '.join(server_cmd)}")
    server_proc = _spawn(server_cmd, server_log)
    client_procs: List[subprocess.Popen] = []
    try:
        _wait_for_server_port(SERVER_HOST, SERVER_PORT, timeout=args.server_start_timeout)
        for cid in range(args.num_clients):
            client_cmd = [
                PYTHON, "client.py",
                "--client-id", str(cid),
                "--num-clients", str(args.num_clients),
                *_dataset_flags(args),
                *_partition_flags(args, point),
                *_partition_seed_flags(args),
                "--training-trace", args.training_trace,
            ]
            client_log = log_dir / f"client_{cid}_baseline_{tag}.log"
            client_procs.append(_spawn(client_cmd, client_log))

        rc = server_proc.wait()
        _close_log(server_proc)
        for p in client_procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _kill(p)
            _close_log(p)
        return rc
    except BaseException:
        _kill(server_proc)
        for p in client_procs:
            _kill(p)
        raise


def _spawn_retrain_baseline(args, point, log_dir: Path) -> int:
    """Spawn retrain_baseline.py with --run-mia chained on. Return process rc.

    Note: retrain_baseline.py's auto-default for --mia-member-source-db is the
    canonical torchvision dataset name (e.g. 'CIFAR10'), which is wrong under
    partition modes that produce suffixed baseline DBs. We override explicitly
    here so the retrain MIA sources c0's sample IDs from the matching
    baseline-DB partition.
    """
    tag = _point_tag(args, point)
    mia_output = args.results_dir / f"mia_phase{args.phase_label}_{tag}_retrain.json"
    mia_plot = args.results_dir / f"mia_phase{args.phase_label}_{tag}_retrain.png"
    baseline_db = _db_name_baseline(args, point)
    cmd = [
        PYTHON, "retrain_baseline.py",
        *_dataset_flags(args),
        *_partition_flags(args, point),
        *_partition_seed_flags(args),  # retrain_baseline.py forwards this to its clients
        "--num-clients", str(args.num_clients),
        "--num-rounds", str(args.num_rounds),
        "--exclude-client", str(args.target_client),
        "--training-trace", args.training_trace,
        "--run-mia",
        "--target-client", str(args.target_client),
        "--mia-attacks", args.mia_attacks,
        "--mia-rounds", str(args.num_rounds),
        "--mia-n-members", str(args.mia_n_members),
        "--mia-n-nonmembers", str(args.mia_n_nonmembers),
        "--mia-non-member-source", args.mia_non_member_source,
        "--mia-member-source-db", baseline_db,
        "--mia-output", str(mia_output),
        "--mia-plot", str(mia_plot),
    ]
    retrain_log = log_dir / f"retrain_{tag}.log"
    print(f"[phase3] Retrain+MIA: {' '.join(cmd)}")
    proc = _spawn(cmd, retrain_log)
    rc = proc.wait()
    _close_log(proc)
    return rc


def _spawn_phase_d(args, point, algos: List[str], log_dir: Path) -> int:
    """Spawn run_phase_d.py for the given algos against the suffixed source DB."""
    tag = _point_tag(args, point)
    source_db_name = _db_name_baseline(args, point)
    results_subdir = args.results_dir / f"point_{tag}"
    cmd = [
        PYTHON, "run_phase_d.py",
        "--dataset", args.dataset,
        "--source-db-name", source_db_name,
        "--model", args.model,
        "--num-clients", str(args.num_clients),
        "--target-client", str(args.target_client),
        "--algorithms", ",".join(algos),
        "--mia-attacks", args.mia_attacks,
        "--mia-rounds", str(args.num_rounds),
        "--mia-n-members", str(args.mia_n_members),
        "--mia-n-nonmembers", str(args.mia_n_nonmembers),
        "--mia-non-member-source", args.mia_non_member_source,
        "--results-dir", str(results_subdir),
    ]
    phase_d_log = log_dir / f"phase_d_{tag}.log"
    print(f"[phase3] Phase-D unlearning+MIA: {' '.join(cmd)}")
    proc = _spawn(cmd, phase_d_log)
    rc = proc.wait()
    _close_log(proc)
    return rc


# -----------------------------------------------------------------------------
# Result extraction
# -----------------------------------------------------------------------------

def _extract_auc(mia_json: Path, target: str, attack: str, round_num: int) -> Optional[float]:
    """Pull members_vs_nonmembers AUC from a per-round MIA JSON."""
    if not mia_json.is_file():
        return None
    with open(mia_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    round_str = str(round_num)
    round_block = data.get("results", {}).get(round_str)
    if not round_block:
        return None
    target_block = round_block.get(target)
    if not target_block:
        return None
    attack_block = target_block.get(attack, {})
    mvn = attack_block.get("members_vs_nonmembers", {})
    return mvn.get("auc")


def _read_baseline_test_acc(db_name: str, round_num: int) -> Optional[Tuple[float, float]]:
    """Read the FL aggregate's metadata (oldest mtime = original) to get test
    accuracy / loss at ``round_num``. Returns (accuracy, loss) or None."""
    round_dir = Path("contributions") / db_name / f"round_{round_num:04d}"
    if not round_dir.is_dir():
        return None
    metas = sorted(
        round_dir.glob("*_aggregated_*_metadata.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not metas:
        return None
    with open(metas[0], "r", encoding="utf-8") as f:  # oldest = original FL aggregate
        meta = json.load(f)
    metrics = meta.get("metrics") or {}
    acc = metrics.get("accuracy")
    loss = metrics.get("loss")
    if acc is None or loss is None:
        return None
    return float(acc), float(loss)


def _compute_test_acc_for_db(args, db_name: str, round_num: int, device: torch.device) -> Optional[Tuple[float, float]]:
    """Load the NEWEST aggregate (post-unlearning if present) at ``round_num``
    and evaluate test accuracy via utils.test(). Returns (accuracy, loss) or None."""
    db = ContributionDB(base_dir="contributions", dataset_name=db_name)
    params = db.get_latest_aggregated_parameters(round_num)
    if params is None:
        return None
    net = create_model(
        args.model,
        args.dataset,
        num_classes=args.num_classes,
        num_channels=args.num_channels,
        img_size=args.img_size,
    ).to(device)
    set_parameters(net, params)

    batch_size = 8 if (args.img_size and args.img_size > 224) else 64
    _, testloader = load_data(
        args.dataset,
        num_clients=1,
        batch_size=batch_size,
        dataset_path=args.dataset_path,
        img_size=args.img_size,
        num_channels=args.num_channels,
    )
    loss, acc = test(net, testloader, device)
    return float(acc), float(loss)


_EMPTY_RE = re.compile(r"left clients \[([0-9,\s]+)\] with zero samples")


def _detect_empty_clients(log_path: Path) -> List[int]:
    """Grep the baseline server log for the Dirichlet zero-samples warning."""
    if not log_path.is_file():
        return []
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    match = _EMPTY_RE.search(text)
    if not match:
        return []
    return [int(x.strip()) for x in match.group(1).split(",") if x.strip()]


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def _parse_alpha_grid(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_share_grid(s: str) -> List[Tuple[float, float]]:
    pairs = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise SystemExit(f"--share-grid entry {tok!r} missing primary:secondary separator")
        p, sec = tok.split(":", 1)
        pairs.append((float(p), float(sec)))
    return pairs


def _enumerate_points(args) -> List:
    if args.partition_mode == "dirichlet":
        return _parse_alpha_grid(args.alpha_grid)
    return _parse_share_grid(args.share_grid)


def _plan_per_point(args, point, algos: List[str], sanity_algos: List[str]) -> Dict:
    tag = _point_tag(args, point)
    plan = {
        "point": point,
        "tag": tag,
        "baseline_db": _db_name_baseline(args, point),
        "retrain_db": _db_name_retrain(args, point),
        "unlearned_dbs": {a: _db_name_unlearned(args, point, a) for a in algos},
    }
    if sanity_algos and args.partition_mode == "dirichlet" and float(point) == float(args.sanity_alpha):
        plan["sanity_dbs"] = {a: _db_name_unlearned(args, point, a) for a in sanity_algos}
    return plan


def _process_point(args, point, algos: List[str], sanity_algos: List[str],
                   device: torch.device, log_dir: Path) -> Dict:
    tag = _point_tag(args, point)
    baseline_db = _db_name_baseline(args, point)
    retrain_db = _db_name_retrain(args, point)
    print(f"\n{'#'*70}\n[phase3] Point {tag}  (baseline DB: {baseline_db})\n{'#'*70}")
    out: Dict = {"tag": tag, "point": point}

    # 1) Baseline training (skip if DB already has the final round).
    baseline_final = Path("contributions") / baseline_db / f"round_{args.num_rounds:04d}"
    if args.skip_existing and baseline_final.is_dir():
        print(f"[phase3] Baseline DB already present for {tag}; skipping training.")
    else:
        rc = _spawn_baseline_training(args, point, log_dir)
        if rc != 0:
            out["error"] = f"baseline_training rc={rc}"
            return out

    # Detect empty clients from the baseline server log.
    empty = _detect_empty_clients(log_dir / f"server_baseline_{tag}.log")
    out["empty_clients"] = empty
    if args.target_client in empty:
        out["error"] = f"target_client {args.target_client} has zero samples; skipping"
        return out

    # Pull original test accuracy from metadata.
    orig_acc_loss = _read_baseline_test_acc(baseline_db, args.num_rounds)
    if orig_acc_loss:
        out["test_accuracy_original"] = orig_acc_loss[0]
        out["test_loss_original"] = orig_acc_loss[1]

    # Class-collapse skip for Phase 3.2 extremes (e.g., 100/0/0/0 known to collapse
    # per Information/PARTIAL_COLLAPSE_FIX.md). Heuristic: test_accuracy < 1/num_classes
    # means the model is below random — pruning + retrain MIA on a degenerate model
    # is uninformative, so skip the rest of the pipeline and record the collapse.
    if (
        orig_acc_loss
        and args.num_classes
        and orig_acc_loss[0] < (1.0 / max(args.num_classes, 2))
    ):
        out["collapsed"] = True
        out["error"] = (
            f"baseline test_accuracy={orig_acc_loss[0]:.3f} below random ({1/args.num_classes:.3f}); "
            "skipping retrain + unlearning"
        )
        return out

    # 2) Retrain baseline + MIA.
    retrain_final = Path("contributions") / retrain_db / f"round_{args.num_rounds:04d}"
    if args.skip_existing and retrain_final.is_dir():
        print(f"[phase3] Retrain DB already present for {tag}; skipping retrain.")
    else:
        rc = _spawn_retrain_baseline(args, point, log_dir)
        if rc != 0:
            out["error"] = f"retrain rc={rc}"
            return out

    retrain_acc_loss = _read_baseline_test_acc(retrain_db, args.num_rounds)
    if retrain_acc_loss:
        out["test_accuracy_retrain"] = retrain_acc_loss[0]
        out["test_loss_retrain"] = retrain_acc_loss[1]

    retrain_mia_path = args.results_dir / f"mia_phase{args.phase_label}_{tag}_retrain.json"
    out["retrain"] = {
        attack: _extract_auc(retrain_mia_path, "original", attack, args.num_rounds)
        for attack in args.mia_attacks.split(",")
    }

    # 3) Per-algorithm unlearning + MIA (curated algos always; sanity algos only at sanity-alpha).
    point_algos = list(algos)
    if sanity_algos and args.partition_mode == "dirichlet" and float(point) == float(args.sanity_alpha):
        point_algos += [a for a in sanity_algos if a not in point_algos]
    rc = _spawn_phase_d(args, point, point_algos, log_dir)
    if rc != 0:
        out["error"] = f"phase_d rc={rc}"
        # Don't return - partial results may still be parseable below.

    algo_results: Dict[str, Dict] = {}
    results_subdir = args.results_dir / f"point_{tag}"
    original_aucs: Dict[str, Optional[float]] = {}
    for algo in point_algos:
        mia_path = results_subdir / f"mia_unlearned_{algo}.json"
        algo_aucs = {
            attack: _extract_auc(mia_path, "unlearned", attack, args.num_rounds)
            for attack in args.mia_attacks.split(",")
        }
        # Pick up FL-baseline AUCs from the same JSON (run_phase_d uses target=both).
        for attack in args.mia_attacks.split(","):
            if original_aucs.get(attack) is None:
                original_aucs[attack] = _extract_auc(mia_path, "original", attack, args.num_rounds)
        unlearned_db = _db_name_unlearned(args, point, algo)
        acc_loss = _compute_test_acc_for_db(args, unlearned_db, args.num_rounds, device)
        if acc_loss:
            algo_aucs["test_accuracy"] = acc_loss[0]
            algo_aucs["test_loss"] = acc_loss[1]
        # Gap vs retrain.
        for attack in args.mia_attacks.split(","):
            r = out["retrain"].get(attack)
            u = algo_aucs.get(attack)
            algo_aucs[f"gap_{attack}"] = (u - r) if (r is not None and u is not None) else None
        algo_results[algo] = algo_aucs

    out["original"] = original_aucs
    out["algos"] = algo_results
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 3.1 / 3.2 heterogeneity sweep orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Partition mode
    p.add_argument("--partition-mode", choices=["dirichlet", "class_vertical"], required=True)
    p.add_argument("--alpha-grid", default=DEFAULT_ALPHA_GRID,
                   help="Comma-separated alpha values (partition-mode=dirichlet).")
    p.add_argument("--share-grid", default=DEFAULT_SHARE_GRID,
                   help="Comma-separated primary:secondary pairs (partition-mode=class_vertical).")
    p.add_argument("--sanity-algos", default="",
                   help="Comma-separated extra algos to run only at --sanity-alpha (dirichlet only).")
    p.add_argument("--sanity-alpha", type=float, default=1000.0,
                   help="alpha value at which to also run --sanity-algos.")

    # Dataset / model
    p.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "CIFAR10", "FashionMNIST", "CUSTOM"])
    p.add_argument("--model", default="resnet18")
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--num-channels", type=int, default=None)
    p.add_argument("--num-classes", type=int, default=None)

    # FL setup
    p.add_argument("--num-clients", type=int, default=4)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--partition-seed", type=int, default=42)
    p.add_argument("--training-trace", default="loss", choices=["none", "loss", "full"])
    p.add_argument("--target-client", type=int, default=0)

    # Algorithms
    p.add_argument("--algorithms", default=DEFAULT_ALGOS,
                   help="Comma-separated algorithm tags (must exist in run_phase_d.ALGO_CONFIGS).")

    # MIA
    p.add_argument("--mia-attacks", default="loss,modent")
    p.add_argument("--mia-n-members", type=int, default=1000)
    p.add_argument("--mia-n-nonmembers", type=int, default=1000)
    p.add_argument("--mia-non-member-source", default="test", choices=["test", "train_pool"])

    # I/O
    p.add_argument("--results-dir", type=Path, default=Path("results/phase31"))
    p.add_argument("--summary", default="phase31_summary.json")
    p.add_argument("--log-dir", type=Path, default=Path("logs/phase31"))
    p.add_argument("--phase-label", default="31",
                   help="Suffix used in MIA filenames (e.g. '31' or '32').")
    p.add_argument("--server-start-timeout", type=float, default=60.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="If a baseline / retrain DB's final round dir already exists, reuse it.")
    return p.parse_args()


def main():
    args = parse_args()
    args.results_dir = Path(args.results_dir)
    args.log_dir = Path(args.log_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    points = _enumerate_points(args)
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    sanity_algos = [a.strip() for a in args.sanity_algos.split(",") if a.strip()]

    print(f"[phase3] Partition: {args.partition_mode}")
    print(f"[phase3] Points: {points}")
    print(f"[phase3] Algorithms: {algos}")
    if sanity_algos:
        print(f"[phase3] Sanity algos (at alpha={args.sanity_alpha} only): {sanity_algos}")

    if args.dry_run:
        plan = [_plan_per_point(args, pt, algos, sanity_algos) for pt in points]
        print(json.dumps(plan, indent=2, default=str))
        print("[phase3] --dry-run: exiting before any subprocesses.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary: Dict = {
        "partition_mode": args.partition_mode,
        "dataset": args.dataset,
        "model": args.model,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "partition_seed": args.partition_seed,
        "target_client": args.target_client,
        "algorithms": algos,
        "sanity_algos": sanity_algos,
        "sanity_alpha": args.sanity_alpha,
        "mia_attacks": args.mia_attacks.split(","),
        "points": [],
    }

    for point in points:
        t0 = time.monotonic()
        per_point = _process_point(args, point, algos, sanity_algos, device, args.log_dir)
        per_point["elapsed_seconds"] = time.monotonic() - t0
        summary["points"].append(per_point)
        # Incremental write so a crash mid-sweep still leaves partial results.
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[phase3] Point {per_point['tag']} done in {per_point['elapsed_seconds']/60:.1f} min")

    print(f"\n[phase3] Summary written to {args.summary}")


if __name__ == "__main__":
    main()
