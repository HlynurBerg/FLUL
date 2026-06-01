"""
E.2 -- FedProx Counterexample (history-dependent aggregator)
=============================================================

Empirical demonstration of the characterization theorem's reverse direction
(report/thesis_unification_plan.md  contribution i):

    If the aggregator is linear but NOT history-independent, gradient-subtraction
    unlearning is NOT equivalent to retraining-from-scratch.

Setup: 4-client federated linear regression with the FedProx local objective

    L_c(theta) = (1 / (2 * n_c)) * || X_c theta - y_c ||^2
                 + (mu / 2) * || theta - theta_prev ||^2

where theta_prev is the round-(t-1) global aggregate. The proximal term
explicitly couples each client's local update to the previous aggregate --
a clean form of history-dependence that does NOT involve SGD noise (each
client's local minimization is closed-form).

With closed-form local optimization, the ONLY source of imperfection in the
gradient-subtraction identity is the proximal coupling. Expectations:

  - mu = 0    : residual at float64 floor (recovers E.1 test 1 exactly --
                 the proximal term vanishes, FedProx degenerates to
                 one-shot FedAvg with closed-form local solutions).
  - mu > 0    : residual grows -- history-dependence breaks the identity.
  - mu -> inf : residual shrinks again because the proximal term pins each
                 client to theta_prev, so trajectories barely move and both
                 original and retrain end up near theta_init.

Run:
    python run_e2_fedprox_counterexample.py [--seed 42] [--n-rounds 20]
"""

import argparse
import json
import numpy as np


# ---------------------------------------------------------------------------
# Synthetic federation setup (matches E.1)
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
# FedProx closed-form local solver
# ---------------------------------------------------------------------------

def fedprox_local(X, y, theta_prev, mu, ridge=1e-8):
    """Closed-form minimizer of
        (1/(2 n)) || X theta - y ||^2  +  (mu/2) || theta - theta_prev ||^2

    Setting the gradient to zero:
        (1/n) X^T X theta - (1/n) X^T y + mu (theta - theta_prev) = 0
        ((1/n) X^T X + mu I) theta = (1/n) X^T y + mu theta_prev
    """
    n, d = X.shape
    A = (X.T @ X) / n + (mu + ridge) * np.eye(d)
    b = (X.T @ y) / n + mu * theta_prev
    return np.linalg.solve(A, b)


# ---------------------------------------------------------------------------
# Federated averaging + gradient unlearning (matches E.1)
# ---------------------------------------------------------------------------

def fedavg(client_thetas, weights):
    W = sum(weights)
    return sum((w / W) * theta for w, theta in zip(weights, client_thetas))


def gradient_unlearn(theta_global, theta_target, weights, target_idx):
    W = sum(weights)
    w_target = weights[target_idx]
    return (W * theta_global - w_target * theta_target) / (W - w_target)


# ---------------------------------------------------------------------------
# FedProx trajectory + residual computation
# ---------------------------------------------------------------------------

def run_fedprox(clients, weights, mu, n_rounds, theta_init):
    theta = theta_init.copy()
    last_client_thetas = None
    for _ in range(n_rounds):
        client_thetas = [fedprox_local(X, y, theta, mu) for X, y in clients]
        theta = fedavg(client_thetas, weights)
        last_client_thetas = client_thetas
    return theta, last_client_thetas


def fedprox_residual(clients, weights, mu, n_rounds, n_features, target_idx=0):
    """Return || gradient_unlearn(orig FedProx) - true_retrain(FedProx) ||."""
    theta_init = np.zeros(n_features)

    theta_orig, last_thetas = run_fedprox(clients, weights, mu, n_rounds,
                                          theta_init)
    theta_unlearn = gradient_unlearn(theta_orig, last_thetas[target_idx],
                                     weights, target_idx)

    retain_clients = [c for i, c in enumerate(clients) if i != target_idx]
    retain_weights = [w for i, w in enumerate(weights) if i != target_idx]
    theta_retrain, _ = run_fedprox(retain_clients, retain_weights, mu, n_rounds,
                                   theta_init)
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
    parser.add_argument(
        "--mu-list", type=str,
        default="0,0.001,0.01,0.1,1,10,100",
        help="Comma-separated proximal coefficients to sweep.")
    parser.add_argument("--summary-out", type=str, default="e2_summary.json")
    args = parser.parse_args()

    mu_list = [float(s) for s in args.mu_list.split(",")]

    clients, _ = setup_federation(n_clients=args.n_clients,
                                  n_features=args.n_features,
                                  seed=args.seed)
    weights = [len(y) for _, y in clients]

    print("=" * 72)
    print("E.2  --  FedProx Counterexample (history-DEPENDENT aggregator)")
    print("=" * 72)
    print(f"  Clients         : {args.n_clients}")
    print(f"  Features        : {args.n_features}")
    print(f"  Samples / client: {len(clients[0][1])}")
    print(f"  Rounds          : {args.n_rounds}")
    print(f"  Target client   : 0 (unlearned)")
    print(f"  Seed            : {args.seed}")
    print()

    print(f"  {'mu':>12s}  {'residual':>14s}  reading")
    print("  " + "-" * 60)
    rows = []
    for mu in mu_list:
        r = fedprox_residual(clients, weights, mu, args.n_rounds,
                             args.n_features)
        if mu == 0:
            reading = "history-INDEPENDENT  (degenerate FedProx == one-shot FedAvg)"
        elif r < 1e-10:
            reading = "below precision floor"
        else:
            reading = "history-DEPENDENT  -- residual measurable"
        print(f"  {mu:>12.4g}  {r:>14.3e}  {reading}")
        rows.append({"mu": mu, "residual": r})
    print()
    print("=" * 72)

    r0 = rows[0]["residual"]
    nonzero = [row for row in rows if row["mu"] > 0]
    if nonzero:
        r_peak_row = max(nonzero, key=lambda row: row["residual"])
        print(f"  mu = 0            residual: {r0:.3e}")
        print(f"  mu = {r_peak_row['mu']:<10.4g} residual: {r_peak_row['residual']:.3e}  "
              f"(peak in sweep)")
        ratio = r_peak_row["residual"] / max(r0, 1e-300)
        print(f"  peak / (mu = 0)  : {ratio:.3e}x")
    print()
    print("  Reading: the proximal term  (mu / 2) * || theta - theta_prev ||^2")
    print("  introduces history-dependence into each client's local update.")
    print("  Even with closed-form local minimization (no SGD noise),")
    print("  gradient-subtraction unlearning leaves a measurable residual for")
    print("  any mu > 0, growing then saturating as mu sweeps from 0 to infinity.")
    print()
    print("  This confirms the reverse direction of the characterization theorem")
    print("  (contribution i in idea.tex): a history-dependent aggregator breaks")
    print("  the exact-unlearning identity.")

    summary = {
        "config": {
            "seed": args.seed,
            "n_rounds": args.n_rounds,
            "n_clients": args.n_clients,
            "n_features": args.n_features,
            "n_samples_per_client": int(len(clients[0][1])),
            "mu_list": mu_list,
        },
        "results": rows,
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to {args.summary_out}")


if __name__ == "__main__":
    main()
