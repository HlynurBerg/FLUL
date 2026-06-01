"""
Retrain-baseline wrapper.

Launches an FL training run that omits a target client from aggregation, writing
the resulting contributions to a SEPARATE DB (suffixed via --retrain-suffix).
This is the "gold standard" baseline referenced in Phase 2.2 step 3 of the
research plan: with the same canonical partition (so the remaining shards are
identical to the original run), and just one client absent.

Approach: frozen-shard retrain. We DO NOT use server.py's --exclude-clients
filter; instead we simply do not spawn the excluded client. The horizontal
partition is keyed by partition_seed and num_clients, so the remaining clients
keep their original shards. The resulting per-round aggregates can be loaded
with evaluate_mia_per_round.py to produce the retrained-baseline AUC curve.

Example:
    python retrain_baseline.py --dataset MNIST --num-clients 4 \
        --num-rounds 20 --exclude-client 0 --training-trace loss \
        --run-mia --target-client 0 \
        --output-dir contributions_retrain
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

PYTHON = sys.executable
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080  # matches server.py / client.py hardcoded bind/connect


def _build_partition_flags(args) -> List[str]:
    """Return the partition-related CLI flags to forward to server.py / client.py."""
    flags = ["--partition-type", args.partition_type]
    if args.partition_type == "dirichlet":
        if args.dirichlet_alpha is None:
            raise SystemExit("--dirichlet-alpha is required when --partition-type=dirichlet")
        flags += ["--dirichlet-alpha", str(args.dirichlet_alpha)]
    if args.partition_type == "class_vertical":
        flags += [
            "--primary-share", str(args.primary_share),
            "--secondary-share", str(args.secondary_share),
        ]
    return flags


def _build_dataset_flags(args) -> List[str]:
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


def _wait_for_server_port(host: str, port: int, timeout: float = 60.0) -> None:
    """Poll the server's TCP port until it accepts a connection or we time out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(
        f"Server did not start listening on {host}:{port} within {timeout:.0f}s. "
        "Check the server log for crashes."
    )


def _spawn(cmd: List[str], log_path: Path) -> subprocess.Popen:
    """Spawn a subprocess with stdout+stderr tee'd to a log file. The log file
    handle is attached to the Popen object as ``._log_file`` so callers can
    close it after wait()."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8", buffering=1)
    f.write(f"$ {' '.join(cmd)}\n\n")
    f.flush()
    env = os.environ.copy()
    # Force UTF-8 on Windows so box-drawing diagnostics in logs don't crash
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
    )
    proc._log_file = f  # type: ignore[attr-defined]
    return proc


def _close_log(proc: subprocess.Popen) -> None:
    f = getattr(proc, "_log_file", None)
    if f is not None:
        try:
            f.close()
        except Exception:
            pass


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _close_log(proc)


def run_retrain(args) -> Path:
    """Spawn server + (num_clients - 1) clients and wait for completion."""
    if not (0 <= args.exclude_client < args.num_clients):
        raise SystemExit(
            f"--exclude-client {args.exclude_client} not in [0, {args.num_clients})"
        )
    suffix = args.retrain_suffix or f"RETRAINED_no{args.exclude_client}"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset_flags = _build_dataset_flags(args)
    partition_flags = _build_partition_flags(args)

    # Pass --exclude-clients so server.py drops min_fit_clients to N-1 and
    # doesn't hang waiting for the absent client.
    server_cmd = [
        PYTHON, "server.py",
        *dataset_flags,
        *partition_flags,
        "--num-rounds", str(args.num_rounds),
        "--retrain-suffix", suffix,
        "--exclude-clients", str(args.exclude_client),
    ]
    print(f"[retrain] Server suffix: {suffix}")
    print(f"[retrain] Server cmd: {' '.join(server_cmd)}")
    server_log = log_dir / f"server_{suffix}.log"
    server_proc = _spawn(server_cmd, server_log)

    client_procs: List[subprocess.Popen] = []
    try:
        _wait_for_server_port(SERVER_HOST, SERVER_PORT, timeout=args.server_start_timeout)
        print(f"[retrain] Server listening on {SERVER_HOST}:{SERVER_PORT}")

        client_ids = [i for i in range(args.num_clients) if i != args.exclude_client]
        print(f"[retrain] Spawning clients {client_ids} (excluding {args.exclude_client})")
        for cid in client_ids:
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
            client_log = log_dir / f"client_{cid}_{suffix}.log"
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
                f"[retrain] WARNING: server rc={rc}, client rcs={client_rcs}. "
                f"Check logs in {log_dir}/"
            )
        else:
            print(f"[retrain] Training complete. server rc=0, clients rc=0.")
    except BaseException:
        _kill(server_proc)
        for p in client_procs:
            _kill(p)
        raise

    if args.partition_type == "class_vertical":
        suffix_name = "CLASS_VERTICAL"
        if not (abs(args.primary_share - 0.85) < 1e-9 and abs(args.secondary_share - 0.05) < 1e-9):
            p_tag = f"{args.primary_share:.2f}".rstrip("0").rstrip(".")
            s_tag = f"{args.secondary_share:.2f}".rstrip("0").rstrip(".")
            suffix_name = f"CLASS_VERTICAL_p{p_tag}_s{s_tag}"
        db_dataset_name = f"{args.dataset}_{suffix_name}_{suffix}"
    elif args.partition_type == "dirichlet":
        a_tag = f"{args.dirichlet_alpha:g}".replace(".", "p")
        db_dataset_name = f"{args.dataset}_DIRICHLET_a{a_tag}_{suffix}"
    else:
        db_dataset_name = f"{args.dataset}_{suffix}"

    db_path = Path("contributions") / db_dataset_name
    print(f"[retrain] Contributions DB at: {db_path}")
    return db_path


def run_mia(args, db_dataset_name: str) -> None:
    """Invoke evaluate_mia_per_round.py against the freshly written retrain DB."""
    # Sample-id source for the member set: whichever DB the target client
    # actually participated in. For a retrain that excluded the target client,
    # that's the ORIGINAL DB (--mia-member-source-db); otherwise it's the
    # retrain DB itself. Default mia_member_source_db = args.dataset (the
    # original/unsuffixed DB) when target-client == exclude-client.
    member_src = args.mia_member_source_db or (
        args.dataset if args.target_client == args.exclude_client else db_dataset_name
    )
    cmd = [
        PYTHON, "evaluate_mia_per_round.py",
        "--db-dir", args.mia_db_dir,
        "--dataset", args.dataset,        # canonical torchvision dataset for transforms/loaders
        "--db-name", db_dataset_name,     # the suffixed contributions subdir
        "--member-source-db-name", member_src,
        "--model", args.model,
        "--target-client", str(args.target_client),
        "--target", "original",
        "--attacks", args.mia_attacks,
        "--n-members", str(args.mia_n_members),
        "--n-nonmembers", str(args.mia_n_nonmembers),
        "--non-member-source", args.mia_non_member_source,
        "--seed", str(args.seed),
        "--output", args.mia_output,
    ]
    if args.mia_plot:
        cmd += ["--plot", args.mia_plot]
    if args.mia_rounds:
        cmd += ["--rounds", args.mia_rounds]
    print(f"[retrain] MIA cmd: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"evaluate_mia_per_round.py failed (rc={rc})")


def parse_args():
    p = argparse.ArgumentParser(
        description="Retrain-baseline wrapper: FL run with one client omitted, "
        "writing to a suffixed DB. Optionally runs per-round MIA after.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training config (mirrors server.py / client.py flags)
    p.add_argument("--dataset", default="MNIST", choices=["MNIST", "CIFAR10", "FashionMNIST", "CUSTOM"])
    p.add_argument("--model", default="simplenet")
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--num-channels", type=int, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--num-clients", type=int, default=4, help="Total clients in the canonical partition.")
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--exclude-client", type=int, default=0, help="Client ID to omit from the retrain.")
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
    p.add_argument(
        "--retrain-suffix",
        default=None,
        help="Suffix for the contributions DB. Default: RETRAINED_no<exclude_client>.",
    )
    p.add_argument("--log-dir", default="logs/retrain", help="Where to tee server/client output.")
    p.add_argument("--server-start-timeout", type=float, default=60.0)

    # Optional MIA pass after training
    p.add_argument("--run-mia", action="store_true", help="Run evaluate_mia_per_round.py on the new DB.")
    p.add_argument(
        "--mia-db-dir",
        default="contributions",
        help="Parent of <DATASET>/ subdir for the MIA driver. Should match where ContributionDB writes.",
    )
    p.add_argument("--target-client", type=int, default=None,
                   help="(MIA) Member-set client. Default: same as --exclude-client (so the "
                        "baseline scores leakage of the absent client's data).")
    p.add_argument("--mia-attacks", default="loss,modent")
    p.add_argument("--mia-rounds", default="all")
    p.add_argument("--mia-n-members", type=int, default=1000)
    p.add_argument("--mia-n-nonmembers", type=int, default=1000)
    p.add_argument("--mia-non-member-source", default="test", choices=["test", "train_pool"])
    p.add_argument("--mia-output", default="mia_retrain.json")
    p.add_argument("--mia-plot", default=None)
    p.add_argument(
        "--mia-member-source-db",
        default=None,
        help="DB subdir to read the target client's sample IDs from. Default: "
        "--dataset (original DB) when scoring the absent client (target == exclude); "
        "otherwise the retrain DB itself.",
    )
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    if args.target_client is None:
        args.target_client = args.exclude_client

    db_path = run_retrain(args)

    if args.run_mia:
        # The MIA driver's --dataset arg matches ContributionDB's dataset_name.
        db_dataset_name = db_path.name
        print(f"\n[retrain] Running per-round MIA against {db_dataset_name}")
        run_mia(args, db_dataset_name)


if __name__ == "__main__":
    main()
