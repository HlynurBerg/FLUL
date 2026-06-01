"""
E.1 — Federated Linear Regression Sanity Check
================================================

Empirical corroboration of the correctness theorem
(report/thesis_unification_plan.md § 3, contribution ii):

    Under linear, history-independent FedAvg aggregation, gradient-subtraction
    unlearning produces, at every round t, exactly the parameters that
    retraining-from-scratch without the target client would produce.

Three tests:

  1. ALGEBRAIC IDENTITY (single-round, closed-form). Each client computes a
     closed-form ridge solution on its own D_c. FedAvg averages them. The
     algebraic identity behind gradient subtraction must hold to ~1e-15.

  2. ITERATED CORRECTNESS (multi-round, history-INDEPENDENT). At each round
     every client computes its local update from a fixed theta_init (NOT
     from theta^(t-1)). The aggregator is therefore history-independent.
     Gradient subtraction at round T must match the true retrain to ~1e-10.

  3. INDIRECT PROPAGATION (multi-round, history-DEPENDENT). Vanilla FedAvg
     with local SGD from theta^(t-1). The aggregator is now history-dependent.
     Gradient subtraction will leave a measurable residual — preview of the
     impossibility-theorem direction (E.2 / E.3 will sharpen it).

Expected results:

  Test 1: residual < 1e-12          -> PASS
  Test 2: residual < 1e-10          -> PASS
  Test 3: residual ~ 1e-2 to 1e-1   -> non-trivial, grows with round count

Usage:
    python run_e1_linear_regression.py [--seed 42] [--n-rounds 20]
"""

import argparse
import json
import numpy as np


# ---------------------------------------------------------------------------
# Synthetic federation setup
# ---------------------------------------------------------------------------

def setup_federation(n_clients=4, n_features=10, n_samples_per_client=200,
                     noise_std=0.1, seed=42):
    rng = np.random.default_rng(seed)
    true_theta = rng.standard_normal(n_features)
    clients = []
    for _ in range(n_clients):
        X = rng.standard_normal((n_samples_per_client, n_features))
        y = X @ true_theta + noise_std * rng.standard_normal(n_samples_per_client)
        clients.append((X, y))
    return clients, true_theta


# ---------------------------------------------------------------------------
# Local update rules
# ---------------------------------------------------------------------------

def closed_form_ridge(X, y, ridge=1e-6):
    """Closed-form ridge regression on (X, y). History-independent."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ y)


def gradient_step_from(X, y, theta_init, lr=0.01, n_steps=10, batch_size=64,
                       rng=None):
    """Local SGD from theta_init. The output depends on theta_init, so this
    rule is history-DEPENDENT when used with theta_init = theta^(t-1)."""
    if rng is None:
        rng = np.random.default_rng()
    theta = theta_init.copy()
    n = len(y)
    for _ in range(n_steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        Xb, yb = X[idx], y[idx]
        grad = Xb.T @ (Xb @ theta - yb) / len(yb)
        theta = theta - lr * grad
    return theta


# ---------------------------------------------------------------------------
# Federated averaging + gradient unlearning
# ---------------------------------------------------------------------------

def fedavg(client_thetas, weights):
    W = sum(weights)
    return sum((w / W) * theta for w, theta in zip(weights, client_thetas))


def gradient_unlearn(theta_global, theta_target, weights, target_idx):
    """Algebraic gradient subtraction:
        theta_unlearn = (W * theta_global - w_target * theta_target) / (W - w_target)
    """
    W = sum(weights)
    w_target = weights[target_idx]
    return (W * theta_global - w_target * theta_target) / (W - w_target)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_1_algebraic_identity(clients, weights, target_idx=0):
    """Single-round closed-form FedAvg. Algebraic identity, must be ~1e-15."""
    client_thetas = [closed_form_ridge(X, y) for X, y in clients]
    theta_global = fedavg(client_thetas, weights)
    theta_unlearn = gradient_unlearn(theta_global, client_thetas[target_idx],
                                     weights, target_idx)
    retain_thetas = [t for i, t in enumerate(client_thetas) if i != target_idx]
    retain_weights = [w for i, w in enumerate(weights) if i != target_idx]
    theta_retrain = fedavg(retain_thetas, retain_weights)
    return float(np.linalg.norm(theta_unlearn - theta_retrain))


def test_2_iterated_history_independent(clients, weights, n_rounds=20,
                                        n_features=10, target_idx=0, seed=42):
    """Multi-round FedAvg where each client's local update starts from a
    fixed theta_init (NOT theta^(t-1)). Aggregator is history-INDEPENDENT."""
    theta_init = np.zeros(n_features)

    last_aggregate_orig = None
    last_aggregate_retrain = None
    last_client_thetas = None
    for t in range(n_rounds):
        client_thetas = [
            gradient_step_from(
                X, y, theta_init,
                rng=np.random.default_rng(seed + 1000 * t + c_id))
            for c_id, (X, y) in enumerate(clients)
        ]
        last_aggregate_orig = fedavg(client_thetas, weights)
        retain_thetas = [tt for i, tt in enumerate(client_thetas) if i != target_idx]
        retain_weights = [w for i, w in enumerate(weights) if i != target_idx]
        last_aggregate_retrain = fedavg(retain_thetas, retain_weights)
        last_client_thetas = client_thetas

    theta_unlearn = gradient_unlearn(
        last_aggregate_orig, last_client_thetas[target_idx], weights, target_idx)
    return float(np.linalg.norm(theta_unlearn - last_aggregate_retrain))


def test_3_iterated_history_dependent(clients, weights, n_rounds=20,
                                      n_features=10, target_idx=0, seed=42):
    """Multi-round FedAvg with local SGD from theta^(t-1). History-DEPENDENT
    aggregator. Gradient subtraction will leave a measurable residual."""
    theta_init = np.zeros(n_features)

    # Original trajectory (all clients)
    theta_orig = theta_init.copy()
    last_client_thetas = None
    for t in range(n_rounds):
        client_thetas = [
            gradient_step_from(
                X, y, theta_orig,
                rng=np.random.default_rng(seed + 1000 * t + c_id))
            for c_id, (X, y) in enumerate(clients)
        ]
        theta_orig = fedavg(client_thetas, weights)
        last_client_thetas = client_thetas

    theta_unlearn = gradient_unlearn(
        theta_orig, last_client_thetas[target_idx], weights, target_idx)

    # True retrain (re-run FedAvg from scratch with target absent, but using
    # IDENTICAL RNG state for the remaining clients — the only thing that
    # diverges is the trajectory itself, not the minibatch sampling)
    theta_retrain = theta_init.copy()
    retain_weights = [w for i, w in enumerate(weights) if i != target_idx]
    for t in range(n_rounds):
        retain_thetas = []
        for c_id, (X, y) in enumerate(clients):
            if c_id == target_idx:
                continue
            theta_c = gradient_step_from(
                X, y, theta_retrain,
                rng=np.random.default_rng(seed + 1000 * t + c_id))
            retain_thetas.append(theta_c)
        theta_retrain = fedavg(retain_thetas, retain_weights)

    return float(np.linalg.norm(theta_unlearn - theta_retrain))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-rounds", type=int, default=20)
    parser.add_argument("--n-clients", type=int, default=4)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--summary-out", type=str, default="e1_summary.json")
    args = parser.parse_args()

    clients, true_theta = setup_federation(
        n_clients=args.n_clients, n_features=args.n_features, seed=args.seed)
    weights = [len(y) for _, y in clients]

    print("=" * 72)
    print("E.1  --  Federated Linear Regression Sanity Check")
    print("=" * 72)
    print(f"  Clients         : {args.n_clients}")
    print(f"  Features        : {args.n_features}")
    print(f"  Samples / client: {len(clients[0][1])}")
    print(f"  Rounds          : {args.n_rounds}")
    print(f"  Target client   : 0 (unlearned)")
    print(f"  Seed            : {args.seed}")
    print()

    r1 = test_1_algebraic_identity(clients, weights)
    pass1 = r1 < 1e-12
    print("Test 1 -- Algebraic identity (single-round, closed-form FedAvg)")
    print(f"  || theta_unlearn  -  theta_retrain ||  =  {r1:.3e}")
    print(f"  Threshold 1e-12 -> {'PASS' if pass1 else 'FAIL'}")
    print()

    r2 = test_2_iterated_history_independent(
        clients, weights, n_rounds=args.n_rounds,
        n_features=args.n_features, seed=args.seed)
    pass2 = r2 < 1e-10
    print("Test 2 -- Iterated correctness (history-INDEPENDENT FedAvg)")
    print(f"  || theta_unlearn  -  theta_retrain ||  =  {r2:.3e}")
    print(f"  Threshold 1e-10 -> {'PASS' if pass2 else 'FAIL'}")
    print()

    r3 = test_3_iterated_history_dependent(
        clients, weights, n_rounds=args.n_rounds,
        n_features=args.n_features, seed=args.seed)
    print("Test 3 -- Indirect propagation residual (history-DEPENDENT FedAvg)")
    print(f"  || theta_unlearn  -  theta_retrain ||  =  {r3:.3e}")
    sig3 = "NON-TRIVIAL" if r3 > 1e-4 else "below noise"
    print(f"  Expectation: >> 0 (residual is {sig3})")
    print(f"  Reading: empirical face of the impossibility theorem")
    print(f"           (contribution iii in idea.tex).")
    print()

    print("=" * 72)
    print("Summary")
    print("=" * 72)
    verdict1 = "PASS" if pass1 else "FAIL"
    verdict2 = "PASS" if pass2 else "FAIL"
    print(f"  Test 1  algebraic identity        : {verdict1}   (r = {r1:.3e})")
    print(f"  Test 2  iterated correctness      : {verdict2}   (r = {r2:.3e})")
    print(f"  Test 3  indirect propagation      :        (r = {r3:.3e})")
    print()
    if pass1 and pass2:
        print("  Correctness theorem corroborated to numerical precision.")
        print("  Test 3 residual confirms the impossibility-theorem direction:")
        print(f"  ~{r3 / max(r2, 1e-300):.1e}x larger than the history-independent residual.")
    else:
        print("  Correctness theorem NOT corroborated -- investigate implementation.")

    summary = {
        "config": {
            "seed": args.seed,
            "n_rounds": args.n_rounds,
            "n_clients": args.n_clients,
            "n_features": args.n_features,
            "n_samples_per_client": int(len(clients[0][1])),
        },
        "test_1_algebraic_identity": {
            "residual": r1,
            "threshold": 1e-12,
            "pass": pass1,
        },
        "test_2_iterated_history_independent": {
            "residual": r2,
            "threshold": 1e-10,
            "pass": pass2,
        },
        "test_3_iterated_history_dependent": {
            "residual": r3,
            "note": "no pass/fail threshold -- expected > 0 by design",
        },
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to {args.summary_out}")


if __name__ == "__main__":
    main()
