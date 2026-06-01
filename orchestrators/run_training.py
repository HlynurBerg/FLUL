"""
Launcher for a canonical FL training run (server + N clients, no exclusion).

Mirrors retrain_baseline.py's process-orchestration but spawns ALL clients,
writes to the unsuffixed DB (e.g. ``contributions/MNIST/``), and is used to
produce the original Phase 2.1 baseline that subsequent unlearning runs and
the retrained baseline are compared against.

Example:
    python run_training.py --dataset MNIST --num-clients 4 --num-rounds 20 \
        --training-trace loss
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from retrain_baseline import (
    SERVER_HOST,
    SERVER_PORT,
    _build_dataset_flags,
    _build_partition_flags,
    _close_log,
    _kill,
    _spawn,
    _wait_for_server_port,
)
import subprocess

PYTHON = sys.executable


def parse_args():
    p = argparse.ArgumentParser(
        description="Spawn a canonical FL training run (server + N clients, no exclusion).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="MNIST", choices=["MNIST", "CIFAR10", "FashionMNIST", "CUSTOM"])
    p.add_argument("--model", default="simplenet")
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--num-channels", type=int, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--num-clients", type=int, default=4)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument(
        "--partition-type",
        default="horizontal",
        choices=["horizontal", "vertical", "class_vertical", "dirichlet"],
    )
    p.add_argument("--partition-seed", type=int, default=None)
    p.add_argument("--dirichlet-alpha", type=float, default=None)
    p.add_argument("--primary-share", type=float, default=0.85)
    p.add_argument("--secondary-share", type=float, default=0.05)
    p.add_argument(
        "--training-trace",
        default="loss",
        choices=["none", "loss", "full"],
        help="Default 'loss' so per-sample manifests are written (required for MIA).",
    )
    p.add_argument("--log-dir", default="logs/training")
    p.add_argument("--server-start-timeout", type=float, default=60.0)
    return p.parse_args()


def main():
    args = parse_args()
    dataset_flags = _build_dataset_flags(args)
    partition_flags = _build_partition_flags(args)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    server_cmd = [
        PYTHON, "server.py",
        *dataset_flags,
        *partition_flags,
        "--num-rounds", str(args.num_rounds),
    ]
    print(f"[training] Server cmd: {' '.join(server_cmd)}")
    server_log = log_dir / f"server_{args.dataset}.log"
    server_proc = _spawn(server_cmd, server_log)

    client_procs: List[subprocess.Popen] = []
    try:
        _wait_for_server_port(SERVER_HOST, SERVER_PORT, timeout=args.server_start_timeout)
        print(f"[training] Server listening on {SERVER_HOST}:{SERVER_PORT}")

        for cid in range(args.num_clients):
            client_cmd = [
                PYTHON, "client.py",
                "--client-id", str(cid),
                "--num-clients", str(args.num_clients),
                *dataset_flags,
                *partition_flags,
                "--training-trace", args.training_trace,
            ]
            if args.partition_seed is not None:
                client_cmd += ["--partition-seed", str(args.partition_seed)]
            client_log = log_dir / f"client_{cid}_{args.dataset}.log"
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
                f"[training] WARNING: server rc={rc}, client rcs={client_rcs}. "
                f"Logs in {log_dir}/"
            )
            sys.exit(1)
        print(f"[training] Training complete. server rc=0, clients rc=0.")
    except BaseException:
        _kill(server_proc)
        for p in client_procs:
            _kill(p)
        raise

    print(f"[training] Contributions DB at: contributions/{args.dataset}")


if __name__ == "__main__":
    main()
