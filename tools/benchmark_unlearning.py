"""
Benchmark unlearning algorithms (gradient, influence, hessian, class_pruning,
gradient_ascent_kd) against a freshly-trained FL run.

Each algorithm is applied to an independent copy of the same FL contributions
database (so they don't interfere). For each algorithm the script measures:

  - test_acc        — utility on the held-out test split
  - retain_acc      — accuracy on retained training samples (the rest of the
                      member pool, not the forgotten ones)
  - forget_acc      — accuracy on forgotten samples (lower = more effective)
  - param_l2_shift  — L2 distance from the original FL aggregate
  - mia_auc         — membership-inference AUC on forgotten samples for
                      loss / modent / lira attacks (lower = more privacy)
  - wall_clock_s    — wall-clock time for the unlearning step

Outputs:
  - JSON report (default ./benchmark_results.json)
  - PNG plot (default ./visualizations/unlearning_benchmark.png)

Usage:
    python benchmark_unlearning.py --dataset MNIST --num-clients 2 \\
        --num-rounds 4 --samples-to-forget 50 --seed 42

Notes:
  * 'gradient' is a client-LEVEL baseline: when included, it removes the ENTIRE
    affected client (not just the forget samples). This is documented in the
    output and visualisation labels.
  * The MIA shadow cache (mia_cache/shadow/) is shared across algorithm runs;
    a single training cost amortises over all of them.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from contributions_db import ContributionDB
from manage_withdrawals import unlearn_client, unlearn_samples_all_rounds
from mia import (
    evaluate_threshold_attack,
    forward_with_ids,
    lira_offline_scores,
    loss_attack_scores,
    modified_entropy_attack_scores,
)
from mia_eval_sets import build_evaluation_loader, sample_train_pool_non_members
from mia_shadow import ShadowSpec, compute_shadow_signal_matrix, train_shadow_models
from model import SimpleNet, get_parameters, set_parameters
from sample_ids import get_global_index_map
from utils import load_data, test, train_epoch


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _fedavg(contribs: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    total = sum(n for _, n in contribs)
    out = None
    for params, n in contribs:
        scaled = [p * (n / total) for p in params]
        if out is None:
            out = scaled
        else:
            for i in range(len(out)):
                out[i] += scaled[i]
    return out


def _l2_dist(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    return float(sum(np.sum((np.asarray(x) - np.asarray(y)) ** 2) for x, y in zip(a, b)) ** 0.5)


def _final_aggregate(db: ContributionDB) -> Tuple[List[np.ndarray], int]:
    stats = db.get_statistics()
    final_round = max(stats["rounds"].keys())
    round_dir = db.contributions_dir / f"round_{final_round:04d}"
    files = sorted(
        round_dir.glob("round_*_aggregated_*_params.pkl"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"No aggregate found in {round_dir}")
    with open(files[-1], "rb") as f:
        return pickle.load(f), final_round


def _original_aggregate(db: ContributionDB, round_num: int) -> List[np.ndarray]:
    """Oldest-mtime aggregate file in the round dir = original FL aggregate."""
    round_dir = db.contributions_dir / f"round_{round_num:04d}"
    files = sorted(
        round_dir.glob("round_*_aggregated_*_params.pkl"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"No aggregate found in {round_dir}")
    with open(files[0], "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# FL simulation (mirrors run_sample_unlearning_demo.py)
# ──────────────────────────────────────────────────────────────────────────────
def run_fl_simulation(
    dataset: str,
    num_clients: int,
    num_rounds: int,
    batch_size: int,
    db: ContributionDB,
    device: torch.device,
) -> Tuple[torch.utils.data.DataLoader, Dict[int, List[int]]]:
    client_loaders, testloader = load_data(
        dataset, num_clients=num_clients, batch_size=batch_size,
    )
    global_net = SimpleNet(num_classes=10, num_channels=1, img_size=28).to(device)
    global_params = get_parameters(global_net)
    db.save_initial_global_parameters(global_params)
    sample_ids_by_client: Dict[int, List[int]] = {}

    for rnd in range(num_rounds):
        contribs = []
        for cid in range(num_clients):
            net = SimpleNet(num_classes=10, num_channels=1, img_size=28).to(device)
            set_parameters(net, global_params)
            sids_acc: List[int] = []
            stats: dict = {}
            train_epoch(
                net, client_loaders[cid], device,
                epochs=1, label_smoothing=0.0,
                sample_ids_accumulator=sids_acc, training_stats=stats,
            )
            params = get_parameters(net)
            n_samples = len(client_loaders[cid].dataset)
            global_ids = get_global_index_map(client_loaders[cid].dataset)
            metrics = {
                "sample_ids": global_ids,
                "training_trace": stats,
                "training_trace_version": 1,
                "sample_id_scheme_version": 1,
            }
            db.save_contribution(
                round_num=rnd, client_id=cid, parameters=params,
                num_samples=n_samples, metrics=metrics,
            )
            contribs.append((params, n_samples))
            if rnd == 0:
                sample_ids_by_client[cid] = list(global_ids)

        global_params = _fedavg(contribs)
        set_parameters(global_net, global_params)
        loss, acc = test(global_net, testloader, device)
        db.save_aggregated_model(
            round_num=rnd, parameters=global_params,
            metrics={"loss": loss, "accuracy": acc},
        )
    return testloader, sample_ids_by_client


# ──────────────────────────────────────────────────────────────────────────────
# Per-algorithm runner
# ──────────────────────────────────────────────────────────────────────────────
def run_algorithm(
    algo: str,
    db: ContributionDB,
    dataset: str,
    client_id: int,
    sample_ids: List[int],
    num_clients: int,
    hp: dict,
) -> float:
    """Apply ``algo`` to ``db`` (in-place). Returns wall-clock seconds."""
    # Withdraw samples in this DB copy so the unlearners see the withdrawal
    ck = db.resolve_contribution_key(0, client_id)
    db.withdraw_samples_from_contribution(ck, sample_ids, reason=f"benchmark_{algo}")

    t0 = time.time()
    if algo == "gradient":
        # Client-level only: removes the ENTIRE client's contribution.
        unlearn_client(
            db=db, dataset_name=dataset, client_id=client_id,
            algorithm="gradient", propagate=True, evaluate=False,
            num_clients=num_clients,
        )
    else:
        # propagate=True is safe after the min→max fix in _propagate_sample_unlearning
        # (call sites in unlearning.py): rounds in sample_rounds keep their per-round
        # aggregate corrections; only rounds AFTER max(sample_rounds) get
        # FedAvg-recalculated to honour any sample-withdrawal weight reductions.
        unlearn_samples_all_rounds(
            db=db, dataset_name=dataset, client_id=client_id,
            sample_ids=sample_ids, algorithm=algo, propagate=True,
            num_clients=num_clients,
            damping_factor=hp.get(f"{algo}_damping", 0.01 if algo == "influence" else 0.1),
            influence_scale=hp.get(f"{algo}_influence_scale", 1.0),
            cg_max_iter=hp.get("cg_max_iter", 30),
            cg_tol=hp.get("cg_tol", 1e-4),
            prune_ratio=hp.get("prune_ratio", 0.1),
            n_probe_per_class=hp.get("n_probe_per_class", 64),
            finetune_epochs=hp.get("finetune_epochs", 0),
            finetune_lr=hp.get("finetune_lr", 1e-3),
            ga_epochs=hp.get("ga_epochs", 3),
            alpha_retain=hp.get("alpha_retain", 1.0),
            gamma_kd=hp.get("gamma_kd", 1.0),
            beta_forget=hp.get("beta_forget", 1.0),
            kd_temperature=hp.get("kd_temperature", 4.0),
            max_forget_steps=hp.get("max_forget_steps", None),
        )
    return time.time() - t0


# ──────────────────────────────────────────────────────────────────────────────
# MIA evaluation
# ──────────────────────────────────────────────────────────────────────────────
def compute_mia_aucs(
    net: torch.nn.Module,
    params: List[np.ndarray],
    forgotten_loader,
    nonmember_loader,
    member_loader,
    shadow_signals: Dict[int, List[float]],
    device: torch.device,
) -> Dict[str, float]:
    set_parameters(net, params)
    md = forward_with_ids(net, member_loader, device) if member_loader is not None else {}
    fd = forward_with_ids(net, forgotten_loader, device) if forgotten_loader is not None else {}
    nd = forward_with_ids(net, nonmember_loader, device)

    out: Dict[str, float] = {}
    for name, fn in [("loss", loss_attack_scores), ("modent", modified_entropy_attack_scores)]:
        _, nm = fn(nd)
        if fd:
            _, f = fn(fd)
            out[name] = float(evaluate_threshold_attack(f, nm).auc) if f.size and nm.size else float("nan")
        else:
            out[name] = float("nan")

    target = {sid: d["scaled_logit"] for sid, d in nd.items()}
    target.update({sid: d["scaled_logit"] for sid, d in md.items()})
    target.update({sid: d["scaled_logit"] for sid, d in fd.items()})
    ids, scores = lira_offline_scores(target, shadow_signals)
    id2s = dict(zip(ids.tolist(), scores.tolist()))
    f_l = np.asarray([id2s[s] for s in fd if s in id2s])
    nm_l = np.asarray([id2s[s] for s in nd if s in id2s])
    out["lira"] = (
        float(evaluate_threshold_attack(f_l, nm_l).auc)
        if f_l.size and nm_l.size
        else float("nan")
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def render_plot(results: Dict[str, dict], config: dict, out_path: Path) -> None:
    algos = list(results.keys())
    panels = [
        ("test_acc", "Test accuracy (utility)", lambda r: r.get("test_acc", float("nan"))),
        ("retain_acc", "Retain accuracy (utility)", lambda r: r.get("retain_acc", float("nan"))),
        ("forget_acc", "Forget accuracy (lower → more effective)", lambda r: r.get("forget_acc", float("nan"))),
        ("mia_loss", "MIA AUC: loss attack (↓ better privacy)", lambda r: r.get("mia_auc", {}).get("loss", float("nan"))),
        ("mia_modent", "MIA AUC: modent attack (↓ better privacy)", lambda r: r.get("mia_auc", {}).get("modent", float("nan"))),
        ("mia_lira", "MIA AUC: LiRA attack (↓ better privacy)", lambda r: r.get("mia_auc", {}).get("lira", float("nan"))),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Unlearning Algorithm Benchmark — {config['dataset']} | "
        f"{config['num_clients']} clients × {config['num_rounds']} rounds | "
        f"{config['samples_to_forget']} samples forgotten",
        fontsize=11,
    )
    flat = axes.flatten()
    colors = plt.get_cmap("tab10").colors

    for ax, (key, title, getter) in zip(flat, panels):
        values = [getter(results[a]) for a in algos]
        bar_colors = [colors[i % len(colors)] for i in range(len(algos))]
        bars = ax.bar(range(len(algos)), values, color=bar_colors)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        if key.startswith("mia_"):
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="random (0.5)")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8, loc="upper right")
        elif key.endswith("_acc"):
            ax.set_ylim(0, 1)
        # Value labels
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default="MNIST", choices=["MNIST", "CIFAR10", "FASHIONMNIST"])
    p.add_argument("--num-clients", type=int, default=2)
    p.add_argument("--num-rounds", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples-to-forget", type=int, default=50)
    p.add_argument("--forget-client", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--algorithms",
        default="influence,hessian,class_pruning,gradient_ascent_kd",
        help="Comma-separated. Add 'gradient' to include the (client-level only) baseline.",
    )
    p.add_argument("--n-mia-members", type=int, default=100)
    p.add_argument("--n-mia-nonmembers", type=int, default=500)
    p.add_argument("--n-shadow", type=int, default=10)
    p.add_argument("--shadow-epochs", type=int, default=5)
    p.add_argument("--shadow-cache", default="mia_cache/shadow")
    p.add_argument("--output-json", default="benchmark_results.json")
    p.add_argument("--output-plot", default="visualizations/unlearning_benchmark.png")
    # Algorithm-specific knobs (sensible defaults)
    p.add_argument("--prune-ratio", type=float, default=0.1)
    p.add_argument("--n-probe-per-class", type=int, default=64)
    p.add_argument("--finetune-epochs", type=int, default=0)
    p.add_argument("--finetune-lr", type=float, default=1e-3)
    p.add_argument("--ga-epochs", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    valid = {"gradient", "influence", "hessian", "class_pruning", "gradient_ascent_kd"}
    bad = [a for a in algorithms if a not in valid]
    if bad:
        print(f"Error: unknown algorithm(s): {bad}. Valid: {sorted(valid)}")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("  Unlearning Algorithm Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Algorithms: {algorithms}")
    print(f"Dataset: {args.dataset} | clients={args.num_clients} | rounds={args.num_rounds}")
    print(f"Samples to forget: {args.samples_to_forget} (from client {args.forget_client})")
    print()

    # 1. FL simulation in a fresh tempdir
    base_tmp = Path(tempfile.mkdtemp(prefix="flul_bench_"))
    base_db_dir = base_tmp / "contributions_base"
    base_db = ContributionDB(base_dir=str(base_db_dir), dataset_name=args.dataset)
    print("─" * 60)
    print("  Phase 1: FL simulation")
    print("─" * 60)
    testloader, sample_ids_by_client = run_fl_simulation(
        args.dataset, args.num_clients, args.num_rounds, args.batch_size, base_db, device,
    )

    # 2. Pick forget set
    rng_pick = np.random.default_rng(args.seed)
    pool = sorted(sample_ids_by_client[args.forget_client])
    n_forget = min(args.samples_to_forget, len(pool))
    forget_ids = sorted(int(x) for x in rng_pick.choice(pool, size=n_forget, replace=False))
    print(f"Picked {len(forget_ids)} forget samples from client {args.forget_client}.")

    # 3. Build MIA evaluation loaders + shadow signals (reused across algorithms)
    print()
    print("─" * 60)
    print("  Phase 2: MIA setup (members / forgotten / non-members + shadow models)")
    print("─" * 60)
    all_member_ids: set = set()
    for ids in sample_ids_by_client.values():
        all_member_ids.update(int(s) for s in ids)
    forget_set = set(forget_ids)
    member_pool = sorted(all_member_ids - forget_set)
    n_eval = min(args.n_mia_members, len(member_pool))
    members_eval = sorted(
        int(x) for x in rng.choice(np.asarray(member_pool), size=n_eval, replace=False)
    )
    nonmember_ids = sample_train_pool_non_members(
        args.dataset, exclude_ids=all_member_ids, n=args.n_mia_nonmembers, seed=args.seed
    )

    member_loader = build_evaluation_loader(
        args.dataset, members_eval, train_split=True, batch_size=256
    )
    forgotten_loader = build_evaluation_loader(
        args.dataset, sorted(forget_set), train_split=True, batch_size=256
    )
    nonmember_loader = build_evaluation_loader(
        args.dataset, nonmember_ids, train_split=True, batch_size=256
    )

    # Retain loader for accuracy evaluation: a small subsample of non-forgotten members
    retain_eval_pool = list(member_pool)
    rng_retain = np.random.default_rng(args.seed + 1)
    if retain_eval_pool:
        retain_eval_ids = sorted(
            int(x) for x in rng_retain.choice(
                np.asarray(retain_eval_pool),
                size=min(500, len(retain_eval_pool)),
                replace=False,
            )
        )
        retain_acc_loader = build_evaluation_loader(
            args.dataset, retain_eval_ids, train_split=True, batch_size=256
        )
    else:
        retain_acc_loader = None

    spec = ShadowSpec(
        dataset=args.dataset,
        model_name="simplenet",
        n_shadow=args.n_shadow,
        sampling_rate=0.5,
        epochs=args.shadow_epochs,
        label_smoothing=0.0,
        batch_size=128,
        seed=args.seed,
    )
    print(f"Shadow spec hash: {spec.hash()}  (cache: {args.shadow_cache})")
    shadows = train_shadow_models(spec, Path(args.shadow_cache), device)
    shadow_signals = compute_shadow_signal_matrix(
        shadows, set(members_eval) | forget_set | set(nonmember_ids)
    )

    # 4. Establish a no-unlearning baseline from the base DB (final-round aggregate)
    print()
    print("─" * 60)
    print("  Phase 3: Baseline + per-algorithm evaluation")
    print("─" * 60)

    eval_net = SimpleNet(num_classes=10, num_channels=1, img_size=28).to(device)
    base_final_params, final_round = _final_aggregate(base_db)

    def _eval_params(params: List[np.ndarray]) -> Dict[str, float]:
        set_parameters(eval_net, params)
        test_loss, test_acc = test(eval_net, testloader, device)
        # retain/forget loaders yield 3-tuples — utils.test() handles both shapes
        retain_acc = (
            float(test(eval_net, retain_acc_loader, device)[1])
            if retain_acc_loader is not None
            else float("nan")
        )
        _, forget_acc = test(eval_net, forgotten_loader, device)
        return {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "retain_acc": float(retain_acc),
            "forget_acc": float(forget_acc),
        }

    results: Dict[str, dict] = {}
    base_metrics = _eval_params(base_final_params)
    base_mia = compute_mia_aucs(
        eval_net, base_final_params, forgotten_loader, nonmember_loader,
        member_loader, shadow_signals, device,
    )
    results["no_unlearning"] = {
        **base_metrics,
        "param_l2_shift": 0.0,
        "wall_clock_s": 0.0,
        "mia_auc": base_mia,
    }
    print(
        f"\n[no_unlearning] test={base_metrics['test_acc']:.3f}  "
        f"retain={base_metrics['retain_acc']:.3f}  forget={base_metrics['forget_acc']:.3f}  "
        f"mia loss={base_mia['loss']:.3f}  modent={base_mia['modent']:.3f}  lira={base_mia['lira']:.3f}"
    )

    hp: dict = {
        "prune_ratio": args.prune_ratio,
        "n_probe_per_class": args.n_probe_per_class,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "ga_epochs": args.ga_epochs,
    }

    for algo in algorithms:
        print(f"\n[{algo}] running …")
        algo_db_dir = base_tmp / f"contributions_{algo}"
        shutil.copytree(base_db_dir, algo_db_dir)
        algo_db = ContributionDB(base_dir=str(algo_db_dir), dataset_name=args.dataset)
        try:
            elapsed = run_algorithm(
                algo, algo_db, args.dataset, args.forget_client, forget_ids,
                args.num_clients, hp,
            )
        except Exception as e:
            print(f"[{algo}] FAILED: {e}")
            results[algo] = {"error": str(e), "wall_clock_s": float("nan")}
            continue
        post_params, _ = _final_aggregate(algo_db)
        eval_metrics = _eval_params(post_params)
        mia = compute_mia_aucs(
            eval_net, post_params, forgotten_loader, nonmember_loader,
            member_loader, shadow_signals, device,
        )
        results[algo] = {
            **eval_metrics,
            "param_l2_shift": _l2_dist(base_final_params, post_params),
            "wall_clock_s": elapsed,
            "mia_auc": mia,
        }
        print(
            f"[{algo}] elapsed={elapsed:.1f}s  test={eval_metrics['test_acc']:.3f}  "
            f"retain={eval_metrics['retain_acc']:.3f}  forget={eval_metrics['forget_acc']:.3f}  "
            f"mia loss={mia['loss']:.3f}  modent={mia['modent']:.3f}  lira={mia['lira']:.3f}"
        )

    # 5. Write outputs
    config = {
        "dataset": args.dataset,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "samples_to_forget": args.samples_to_forget,
        "forget_client": args.forget_client,
        "seed": args.seed,
        "n_mia_members": args.n_mia_members,
        "n_mia_nonmembers": args.n_mia_nonmembers,
        "n_shadow": args.n_shadow,
        "shadow_epochs": args.shadow_epochs,
        "hyperparameters": hp,
    }
    output = {
        "config": config,
        "forget_set": {"client_id": args.forget_client, "sample_ids": forget_ids},
        "results": results,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {out_json}")

    out_plot = Path(args.output_plot)
    render_plot(results, config, out_plot)
    print(f"Plot written to:    {out_plot}")

    # 6. Print summary table
    print()
    print("=" * 80)
    print("  Summary")
    print("=" * 80)
    header = f"{'algorithm':<22}{'test':>8}{'retain':>9}{'forget':>9}{'L2':>10}{'time(s)':>10}{'mia_loss':>11}{'mia_modent':>12}{'mia_lira':>11}"
    print(header)
    print("─" * len(header))
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<22}  ERROR: {r['error']}")
            continue
        print(
            f"{name:<22}"
            f"{r['test_acc']:>8.3f}"
            f"{r['retain_acc']:>9.3f}"
            f"{r['forget_acc']:>9.3f}"
            f"{r['param_l2_shift']:>10.3f}"
            f"{r['wall_clock_s']:>10.2f}"
            f"{r['mia_auc']['loss']:>11.3f}"
            f"{r['mia_auc']['modent']:>12.3f}"
            f"{r['mia_auc']['lira']:>11.3f}"
        )
    print("=" * 80)
    print()
    print(f"Temp dir: {base_tmp}  (will be cleaned up)")
    shutil.rmtree(base_tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
