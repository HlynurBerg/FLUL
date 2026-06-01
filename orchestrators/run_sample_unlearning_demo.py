"""
Self-contained demo: federated learning simulation + sample-level unlearning.

Simulates 4 FL rounds with 2 clients (SimpleNet on MNIST, 500 samples/client),
then withdraws a handful of samples and runs per-sample influence unlearning.
Generates plots comparing:
  1. Test accuracy per round (original vs post-unlearning aggregate)
  2. L2 parameter shift per round caused by unlearning
  3. Weight distribution before / after unlearning (first FC layer)

Usage:
    source venv/Scripts/activate && python run_sample_unlearning_demo.py
"""
import pickle
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from contributions_db import ContributionDB
from manage_withdrawals import unlearn_samples_all_rounds, withdraw_samples
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
from sample_ids import WithSampleIds, get_global_index_map
from utils import load_data, test, train_epoch


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DATASET        = "MNIST"
NUM_CLIENTS    = 2
NUM_ROUNDS     = 4
SAMPLES_TO_FORGET = 5          # how many samples from client 0 to forget
BATCH_SIZE     = 32
SEED           = 42
OUT_DIR        = Path("visualizations")
OUT_DIR.mkdir(exist_ok=True)

# MIA evaluation
MIA_N_MEMBERS     = 100
MIA_N_NONMEMBERS  = 500
MIA_N_SHADOW      = 10
MIA_SHADOW_EPOCHS = 5
MIA_SHADOW_CACHE  = Path("mia_cache/shadow")  # persistent across demo runs


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def fedavg(contributions):
    """Weighted average of [(params, num_samples), ...]."""
    total = sum(n for _, n in contributions)
    result = None
    for params, n in contributions:
        if result is None:
            result = [p * (n / total) for p in params]
        else:
            for i in range(len(result)):
                result[i] += params[i] * (n / total)
    return result


def l2_dist(a, b):
    return float(sum(np.sum((x - y) ** 2) for x, y in zip(a, b)) ** 0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Setup: data, model, DB
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Federated Learning + Sample Unlearning Demo")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Temp dir for this run's contribution DB
tmp_dir = tempfile.mkdtemp(prefix="flul_demo_")
db = ContributionDB(base_dir=str(Path(tmp_dir) / "contributions"), dataset_name=DATASET)

# Load data: 2-client horizontal split (uses same stable seed as production)
client_loaders, testloader = load_data(
    DATASET,
    num_clients=NUM_CLIENTS,
    batch_size=BATCH_SIZE,
)

# Initial global model
global_net = SimpleNet(num_classes=10, num_channels=1, img_size=28).to(device)
global_params = get_parameters(global_net)
db.save_initial_global_parameters(global_params)
print(f"Model: SimpleNet ({sum(p.numel() for p in global_net.parameters()):,} params)")
print(f"Clients: {NUM_CLIENTS}  |  Rounds: {NUM_ROUNDS}  |  Samples to forget: {SAMPLES_TO_FORGET}\n")


# ──────────────────────────────────────────────────────────────────────────────
# FL simulation
# ──────────────────────────────────────────────────────────────────────────────
round_acc      = {}    # round_num -> test accuracy
round_sample_ids_by_client = {}  # populated for round 0 only (for sample selection)

print("─" * 60)
print("  Training")
print("─" * 60)

for rnd in range(NUM_ROUNDS):
    print(f"\nRound {rnd}")
    contributions = []

    for cid in range(NUM_CLIENTS):
        # Each client trains from the current global params
        net = SimpleNet(num_classes=10, num_channels=1, img_size=28).to(device)
        set_parameters(net, global_params)

        loader = client_loaders[cid]
        sids_acc = []
        training_stats = {}

        train_epoch(
            net,
            loader,
            device,
            epochs=1,
            label_smoothing=0.0,
            sample_ids_accumulator=sids_acc,
            training_stats=training_stats,
        )

        params = get_parameters(net)
        n_samples = len(loader.dataset)
        global_ids = get_global_index_map(loader.dataset)

        metrics = {
            "sample_ids": global_ids,
            "training_trace": training_stats,
            "training_trace_version": 1,
            "sample_id_scheme_version": 1,
        }

        db.save_contribution(
            round_num=rnd,
            client_id=cid,
            parameters=params,
            num_samples=n_samples,
            metrics=metrics,
        )
        contributions.append((params, n_samples))
        if rnd == 0:
            round_sample_ids_by_client[cid] = global_ids
        print(f"  Client {cid}: {n_samples} samples trained")

    # FedAvg
    global_params = fedavg(contributions)
    set_parameters(global_net, global_params)

    # Evaluate
    loss, acc = test(global_net, testloader, device)
    round_acc[rnd] = acc
    db.save_aggregated_model(round_num=rnd, parameters=global_params,
                             metrics={"loss": loss, "accuracy": acc})
    print(f"  Aggregate → loss={loss:.4f}  acc={acc:.4f}")

print()


# ──────────────────────────────────────────────────────────────────────────────
# Pick samples to forget (from client 0, round 0)
# ──────────────────────────────────────────────────────────────────────────────
forget_client = 0
# Use the first N sample IDs seen in round 0 for client 0
target_global_ids = sorted(round_sample_ids_by_client[forget_client])[:SAMPLES_TO_FORGET]
print("─" * 60)
print(f"  Withdrawing {SAMPLES_TO_FORGET} samples from client {forget_client}")
print(f"  Sample IDs: {target_global_ids}")
print("─" * 60)

# Register withdrawal in DB (resolve_contribution_key returns the actual string key)
ck_r0 = db.resolve_contribution_key(0, forget_client)
db.withdraw_samples_from_contribution(ck_r0, target_global_ids, reason="demo_forget")


# ──────────────────────────────────────────────────────────────────────────────
# Sample-level unlearning across all rounds
# ──────────────────────────────────────────────────────────────────────────────
print("\nRunning unlearn_samples_all_rounds ...")
results = unlearn_samples_all_rounds(
    db=db,
    dataset_name=DATASET,
    client_id=forget_client,
    sample_ids=target_global_ids,
    algorithm="influence",
    propagate=True,
    influence_scale=1.0,
    damping_factor=0.01,
)

print(f"\nUnlearning result:")
for r in results.get("unlearned_rounds", []):
    status = r["status"]
    rnd_n = r["round"]
    if status == "success":
        m = r["metadata"]
        print(f"  Round {rnd_n}: {status}  weight_source={m.get('weight_source','?')}  "
              f"fraction={m.get('sample_weight_fraction', 0):.4f}")
    else:
        print(f"  Round {rnd_n}: {status}  error={r.get('error')}")


# ──────────────────────────────────────────────────────────────────────────────
# Collect post-unlearning accuracy and parameter shifts per round
# ──────────────────────────────────────────────────────────────────────────────
round_acc_unlearned = {}
param_shift = {}           # round_num -> L2 distance original vs unlearned

for rnd in range(NUM_ROUNDS):
    round_dir = Path(tmp_dir) / "contributions" / DATASET / f"round_{rnd:04d}"

    # Load latest original aggregate (pre-unlearning)
    orig_files = sorted(
        [p for p in round_dir.glob("round_*_aggregated_*_params.pkl")
         if "unlearned" not in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    # Load latest unlearned aggregate
    unl_files = sorted(
        round_dir.glob("round_*_aggregated_*_params.pkl"),
        key=lambda p: p.stat().st_mtime,
    )

    if not orig_files or not unl_files:
        continue

    with open(orig_files[0], "rb") as f:
        orig = pickle.load(f)
    with open(unl_files[-1], "rb") as f:
        unl = pickle.load(f)

    # Accuracy of the unlearned aggregate
    set_parameters(global_net, unl)
    _, acc_unl = test(global_net, testloader, device)
    round_acc_unlearned[rnd] = acc_unl

    param_shift[rnd] = l2_dist(orig, unl)
    print(f"  Round {rnd}: acc_unlearned={acc_unl:.4f}  param_shift_L2={param_shift[rnd]:.6f}")


# ──────────────────────────────────────────────────────────────────────────────
# MIA evaluation: did unlearning reduce the attacker's ability to flag
# forgotten samples as members? Score on the FINAL-round aggregate.
# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 60)
print("  MIA Evaluation")
print("─" * 60)

all_member_ids = set()
for ids in round_sample_ids_by_client.values():
    all_member_ids.update(ids)
forgotten_set = set(target_global_ids)
member_pool = sorted(all_member_ids - forgotten_set)

mia_rng = np.random.default_rng(SEED)
n_eval = min(MIA_N_MEMBERS, len(member_pool))
members_eval = sorted(int(x) for x in mia_rng.choice(member_pool, size=n_eval, replace=False))
nonmember_ids = sample_train_pool_non_members(
    DATASET, exclude_ids=all_member_ids, n=MIA_N_NONMEMBERS, seed=SEED
)
print(
    f"Members: {len(members_eval)} eval (of {len(member_pool)} survivors)  "
    f"Forgotten: {len(forgotten_set)}  Non-members: {len(nonmember_ids)} from train pool"
)

member_loader = build_evaluation_loader(DATASET, members_eval, train_split=True, batch_size=256)
forgotten_loader = build_evaluation_loader(
    DATASET, sorted(forgotten_set), train_split=True, batch_size=256
)
nonmember_loader = build_evaluation_loader(
    DATASET, nonmember_ids, train_split=True, batch_size=256
)

print(f"\nTraining/loading {MIA_N_SHADOW} shadow models for LiRA (cache: {MIA_SHADOW_CACHE})")
shadow_spec = ShadowSpec(
    dataset=DATASET,
    model_name="simplenet",
    n_shadow=MIA_N_SHADOW,
    sampling_rate=0.5,
    epochs=MIA_SHADOW_EPOCHS,
    label_smoothing=0.0,
    batch_size=128,
    seed=SEED,
)
shadows = train_shadow_models(shadow_spec, MIA_SHADOW_CACHE, device)
shadow_signals = compute_shadow_signal_matrix(
    shadows, set(members_eval) | forgotten_set | set(nonmember_ids)
)


def _mia_for_target(net, params, label):
    set_parameters(net, params)
    md = forward_with_ids(net, member_loader, device)
    fd = forward_with_ids(net, forgotten_loader, device)
    nd = forward_with_ids(net, nonmember_loader, device)

    out = {}
    for atk_name, fn in [
        ("loss", loss_attack_scores),
        ("modent", modified_entropy_attack_scores),
    ]:
        _, m = fn(md)
        _, f = fn(fd)
        _, nm = fn(nd)
        out[atk_name] = {
            "members":   evaluate_threshold_attack(m, nm).auc,
            "forgotten": evaluate_threshold_attack(f, nm).auc,
        }

    # Combine target signals into one dict, score with LiRA, then split by group
    target = {sid: d["scaled_logit"] for sid, d in nd.items()}
    target.update({sid: d["scaled_logit"] for sid, d in md.items()})
    target.update({sid: d["scaled_logit"] for sid, d in fd.items()})
    ids, scores = lira_offline_scores(target, shadow_signals)
    id2s = dict(zip(ids.tolist(), scores.tolist()))

    def _scores_for(group):
        return np.asarray([id2s[s] for s in group if s in id2s])

    m_l = _scores_for(md)
    f_l = _scores_for(fd)
    nm_l = _scores_for(nd)
    out["lira"] = {
        "members":   evaluate_threshold_attack(m_l, nm_l).auc if m_l.size and nm_l.size else float("nan"),
        "forgotten": evaluate_threshold_attack(f_l, nm_l).auc if f_l.size and nm_l.size else float("nan"),
    }

    print(f"\n  Target {label}:")
    for atk in ("loss", "modent", "lira"):
        print(
            f"    {atk:7s}  members_AUC={out[atk]['members']:.3f}  "
            f"forgotten_AUC={out[atk]['forgotten']:.3f}"
        )
    return out


# Reload final-round aggregates. Convention: oldest aggregate file = original
# FL aggregate; newest = post-unlearning (unlearning saves a fresh aggregate
# after the FL save, so mtime ordering is reliable).
final_round = NUM_ROUNDS - 1
final_dir = Path(tmp_dir) / "contributions" / DATASET / f"round_{final_round:04d}"
all_final_files = sorted(
    final_dir.glob("round_*_aggregated_*_params.pkl"),
    key=lambda p: p.stat().st_mtime,
)
with open(all_final_files[0], "rb") as f:
    orig_final = pickle.load(f)
with open(all_final_files[-1], "rb") as f:
    unl_final = pickle.load(f)

mia_orig = _mia_for_target(global_net, orig_final, "original")
mia_unl  = _mia_for_target(global_net, unl_final, "unlearned")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
rounds = sorted(round_acc.keys())
acc_orig   = [round_acc[r] for r in rounds]
acc_unl    = [round_acc_unlearned.get(r, float("nan")) for r in rounds]
shifts     = [param_shift.get(r, 0.0) for r in rounds]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    f"Sample-Level Unlearning Demo  |  SimpleNet on {DATASET}  |  "
    f"{NUM_CLIENTS} clients, {NUM_ROUNDS} rounds, {SAMPLES_TO_FORGET} samples forgotten",
    fontsize=11,
)
axes_flat = axes.flatten()

# ── Plot 1: accuracy per round ────────────────────────────────────────────────
ax = axes_flat[0]
ax.plot(rounds, acc_orig, "o-", label="Original aggregate", color="steelblue")
ax.plot(rounds, acc_unl,  "s--", label="After unlearning",  color="tomato")
ax.set_xlabel("Round")
ax.set_ylabel("Test accuracy")
ax.set_title("Test Accuracy per Round")
ax.legend()
ax.set_xticks(rounds)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# ── Plot 2: L2 parameter shift per round ─────────────────────────────────────
ax = axes_flat[1]
bars = ax.bar(rounds, shifts, color="mediumpurple", alpha=0.8)
ax.set_xlabel("Round")
ax.set_ylabel("L2 distance")
ax.set_title("Parameter Shift from Unlearning\n(L2 distance: original vs unlearned)")
ax.set_xticks(rounds)
ax.grid(True, axis="y", alpha=0.3)
for bar, v in zip(bars, shifts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1e-5,
            f"{v:.5f}", ha="center", va="bottom", fontsize=8)

# ── Plot 3: weight distribution before vs after for fc1 ───────────────────────
ax = axes_flat[2]
# Load original round-0 and unlearned round-0 params
round0_dir = Path(tmp_dir) / "contributions" / DATASET / "round_0000"
orig_files = sorted(
    [p for p in round0_dir.glob("round_*_aggregated_*_params.pkl")
     if "unlearned" not in p.name],
    key=lambda p: p.stat().st_mtime,
)
unl_files = sorted(
    round0_dir.glob("round_*_aggregated_*_params.pkl"),
    key=lambda p: p.stat().st_mtime,
)
if orig_files and unl_files:
    with open(orig_files[0], "rb") as f:
        p_orig = pickle.load(f)
    with open(unl_files[-1], "rb") as f:
        p_unl = pickle.load(f)
    # fc1 is the 4th param tensor (index 4 in SimpleNet state_dict order)
    # Find the largest dense weight matrix (fc1.weight)
    fc1_idx = max(range(len(p_orig)), key=lambda i: p_orig[i].size)
    w_orig = p_orig[fc1_idx].flatten()
    w_unl  = p_unl[fc1_idx].flatten()
    ax.hist(w_orig, bins=60, alpha=0.5, label="Original",    color="steelblue",  density=True)
    ax.hist(w_unl,  bins=60, alpha=0.5, label="After unlearn", color="tomato", density=True)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Density")
    ax.set_title("Largest FC Weight Distribution\n(Round 0 aggregate)")
    ax.legend()
    ax.grid(True, alpha=0.3)

# ── Plot 4: MIA AUC — leakage on forgotten samples drops after unlearning ─────
ax = axes_flat[3]
attack_names = ["loss", "modent", "lira"]
x_pos = np.arange(len(attack_names))
bar_w = 0.27
ax.bar(
    x_pos - bar_w,
    [mia_orig[a]["members"] for a in attack_names],
    bar_w, label="members (orig, utility)", color="steelblue",
)
ax.bar(
    x_pos,
    [mia_orig[a]["forgotten"] for a in attack_names],
    bar_w, label="forgotten (orig, leakage)", color="tomato",
)
ax.bar(
    x_pos + bar_w,
    [mia_unl[a]["forgotten"] for a in attack_names],
    bar_w, label="forgotten (unlearned)", color="mediumseagreen",
)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="random (0.5)")
ax.set_xticks(x_pos)
ax.set_xticklabels(attack_names)
ax.set_ylabel("MIA AUC")
ax.set_ylim(0, 1)
ax.set_title("MIA leakage on forgotten samples\n(forgotten/orig ↘ forgotten/unlearned)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "sample_unlearning_demo.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Summary")
print("=" * 60)
print(f"  Rounds trained  : {NUM_ROUNDS}")
print(f"  Samples forgotten: {SAMPLES_TO_FORGET} (client {forget_client}, IDs {target_global_ids})")
print(f"  Algorithm        : influence (first-order)")
print()
for r in rounds:
    orig = round_acc.get(r, float("nan"))
    unl  = round_acc_unlearned.get(r, float("nan"))
    shift = param_shift.get(r, 0.0)
    delta = unl - orig if not (np.isnan(orig) or np.isnan(unl)) else float("nan")
    print(f"  Round {r}:  acc_orig={orig:.4f}  acc_unl={unl:.4f}  "
          f"Δacc={delta:+.4f}  L2_shift={shift:.6f}")

print()
print(f"  Contribution DB  : {tmp_dir}")
print("=" * 60)

# Clean up temp dir
shutil.rmtree(tmp_dir, ignore_errors=True)
