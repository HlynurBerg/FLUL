"""
Phase 2.3 (withdrawal-timing) orchestrator.

For each K in {5, 10, 15} and each algo in {class_pruning_p05_ft2, gradient}:
  1. Warm-start a continuation FL run from the post-unlearning round-K aggregate
     in contributions/CIFAR10_UNLEARNED_<algo>/ (newest mtime = post-unlearning).
  2. Train (20-K) more rounds with clients 1/2/3 only (c0 has withdrawn).
  3. Save into contributions/CIFAR10_WD_K<K>_<algo>/, round numbers offset by K
     so the final continuation aggregate is at round 20.

K=20 is the no-op case: the Phase 2.2 unlearned aggregate IS the answer.

Reuses spawn helpers from retrain_baseline.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

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
DEFAULT_NUM_CLIENTS = 4
DEFAULT_TOTAL_ROUNDS = 20
DEFAULT_EXCLUDE_CLIENT = 0
DEFAULT_K_LIST = [5, 10, 15]
DEFAULT_ALGOS = ["class_pruning_p05_ft2", "gradient"]


def _continuation_db_name(dataset: str, k: int, algo: str) -> str:
    return f"{dataset}_WD_K{k:02d}_{algo}"


def _source_db_name(dataset: str, algo: str) -> str:
    return f"{dataset}_UNLEARNED_{algo}"


def _run_one(args, k: int, algo: str) -> Path:
    """Spawn server + 3 clients for the continuation training of (K, algo)."""
    source_db = _source_db_name(args.dataset, algo)
    out_suffix = f"WD_K{k:02d}_{algo}"  # server suffixes as <DATASET>_<suffix>
    continuation_rounds = args.total_rounds - k

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    server_cmd = [
        PYTHON, "server.py",
        "--dataset", args.dataset,
        "--model", args.model,
        "--num-rounds", str(continuation_rounds),
        "--partition-type", "horizontal",
        "--retrain-suffix", out_suffix,
        "--init-weights-from-dataset", source_db,
        "--init-weights-from-round", str(k),
        "--first-round-offset", str(k),
        "--exclude-clients", str(args.exclude_client),
    ]
    print(f"[phase23] K={k} algo={algo}: server cmd = {' '.join(server_cmd)}")
    server_log = log_dir / f"server_{out_suffix}.log"
    server_proc = _spawn(server_cmd, server_log)

    client_procs: List[subprocess.Popen] = []
    try:
        _wait_for_server_port(SERVER_HOST, SERVER_PORT, timeout=args.server_start_timeout)
        print(f"[phase23] Server listening on {SERVER_HOST}:{SERVER_PORT}")

        client_ids = [i for i in range(args.num_clients) if i != args.exclude_client]
        print(f"[phase23] Spawning clients {client_ids} (excluding {args.exclude_client})")
        for cid in client_ids:
            client_cmd = [
                PYTHON, "client.py",
                "--client-id", str(cid),
                "--num-clients", str(args.num_clients),
                "--dataset", args.dataset,
                "--model", args.model,
                "--partition-type", "horizontal",
                "--training-trace", args.training_trace,
            ]
            if args.partition_seed is not None:
                client_cmd += ["--partition-seed", str(args.partition_seed)]
            client_log = log_dir / f"client_{cid}_{out_suffix}.log"
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
            print(
                f"[phase23] WARNING: K={k} algo={algo}: server rc={rc}, client rcs={client_rcs}. "
                f"Check logs in {log_dir}/"
            )
        else:
            print(f"[phase23] K={k} algo={algo}: training complete.")
    except BaseException:
        _kill(server_proc)
        for p in client_procs:
            _kill(p)
        raise

    db_dataset_name = f"{args.dataset}_{out_suffix}"
    db_path = Path("contributions") / db_dataset_name
    print(f"[phase23] K={k} algo={algo}: DB at {db_path}")
    return db_path


def _verify_source(args, k: int, algo: str) -> None:
    src_db = Path("contributions") / _source_db_name(args.dataset, algo)
    round_dir = src_db / f"round_{k:04d}"
    if not round_dir.is_dir():
        raise SystemExit(
            f"[phase23] Source DB missing: {round_dir} does not exist. "
            f"Run Phase 2.2 unlearning for '{algo}' first."
        )
    aggs = list(round_dir.glob("*_aggregated_*_params.pkl"))
    if not aggs:
        raise SystemExit(f"[phase23] No aggregate found in {round_dir}")
    newest = max(aggs, key=lambda p: p.stat().st_mtime)
    print(f"[phase23] Source for K={k} algo={algo}: {newest.name}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2.3: warm-started continuation training after early unlearning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    p.add_argument("--total-rounds", type=int, default=DEFAULT_TOTAL_ROUNDS,
                   help="Final round number T. Continuation runs for (T - K) rounds.")
    p.add_argument("--exclude-client", type=int, default=DEFAULT_EXCLUDE_CLIENT,
                   help="The withdrawn client (default 0).")
    p.add_argument("--k-list", type=str, default=",".join(str(x) for x in DEFAULT_K_LIST),
                   help="Comma-separated withdrawal rounds to sweep.")
    p.add_argument("--algos", type=str, default=",".join(DEFAULT_ALGOS),
                   help="Comma-separated unlearning algorithm tags (must match CIFAR10_UNLEARNED_<algo> DB names).")
    p.add_argument("--partition-seed", type=int, default=None)
    p.add_argument("--training-trace", default="loss", choices=["none", "loss", "full"])
    p.add_argument("--log-dir", default="logs/phase23")
    p.add_argument("--server-start-timeout", type=float, default=60.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Verify source DBs exist and print planned runs; don't spawn anything.")
    return p.parse_args()


def main():
    args = parse_args()
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    print(f"[phase23] Sweep: K={k_list} × algos={algos}")
    print(f"[phase23] Total rounds T={args.total_rounds}, withdrawn client={args.exclude_client}")

    # Pre-flight: verify every source aggregate exists before we burn compute.
    for k in k_list:
        for algo in algos:
            _verify_source(args, k, algo)

    if args.dry_run:
        print("[phase23] --dry-run: source DBs verified, exiting before training.")
        return

    results = []
    for algo in algos:
        for k in k_list:
            t0 = time.monotonic()
            db_path = _run_one(args, k, algo)
            dt = time.monotonic() - t0
            print(f"[phase23] DONE K={k} algo={algo} in {dt/60:.1f} min")
            results.append((k, algo, db_path, dt))

    print("\n[phase23] Summary:")
    for k, algo, db, dt in results:
        print(f"  K={k:02d} algo={algo:<24}  {dt/60:5.1f} min  -> {db}")


if __name__ == "__main__":
    main()
