"""
E.2b -- Median Aggregator Counterexample (non-linear aggregation)
==================================================================

Empirical demonstration of the characterization theorem's
necessity-of-linearity direction (report/theorems_draft.tex Theorem 2):

    If the aggregator is non-linear, no linear unlearning operator with input
    (theta_aggregate, theta_target, w_target, W) can be exact.

Setup: 3-client federated linear regression with closed-form local solutions
(history-INDEPENDENT) and coordinate-wise MEDIAN aggregator. Each client
returns its closed-form ridge solution; the server takes the per-coordinate
median across the 3 clients.

We measure the residual of two natural "unlearning" operators:

  1. Gradient-subtraction formula  Lambda = (W theta_med - w theta_target)
                                            / (W - w_target)
     -- the obvious analog of FedAvg unlearning.
  2. The OLS-optimal linear operator
       Lambda(theta_med, theta_target) = a theta_med + b theta_target
     fitted by least squares over n_trials data instances.

The first operator should leave a residual of order 1 (median loses
information that gradient subtraction cannot recover). The second operator's
residual is the best possible for the class of data-independent linear
unlearning rules over this data distribution; the theorem predicts it is
strictly positive.

Run:
    python run_e2b_median_aggregator.py [--seed 42] [--n-trials 500]
"""

import argparse
import json
import numpy as np


def setup_trial(rng, n_features=5, n_samples=100, noise_std=0.1):
    """Generate one synthetic federation: 3 clients, drawn from independent
    Gaussian-feature regressions with independent ground-truth coefficients
    (so the clients' local solutions disagree)."""
    clients = []
    for _ in range(3):
        true_w = rng.standard_normal(n_features)
        X = rng.standard_normal((n_samples, n_features))
        y = X @ true_w + noise_std * rng.standard_normal(n_samples)
        clients.append((X, y))
    return clients


def closed_form_ridge(X, y, ridge=1e-6):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ y)


def coord_median(thetas):
    """Coordinate-wise median across a list of vectors."""
    stacked = np.stack(thetas, axis=0)
    if stacked.shape[0] == 2:
        return stacked.mean(axis=0)
    return np.median(stacked, axis=0)


def trial_artifacts(clients, target_idx=0):
    """Compute (theta_med, theta_target, theta_retrain) for one trial."""
    client_thetas = [closed_form_ridge(X, y) for X, y in clients]
    theta_med = coord_median(client_thetas)
    theta_target = client_thetas[target_idx]
    retain_thetas = [t for i, t in enumerate(client_thetas) if i != target_idx]
    theta_retrain = coord_median(retain_thetas)
    return theta_med, theta_target, theta_retrain


def gradient_subtraction(theta_med, theta_target, weights, target_idx):
    """The obvious analog of FedAvg unlearning."""
    W = sum(weights)
    w_target = weights[target_idx]
    return (W * theta_med - w_target * theta_target) / (W - w_target)


def fit_optimal_linear(theta_meds, theta_targets, theta_retrains):
    """Find data-independent (a, b) minimizing
        sum_i || a theta_med_i + b theta_target_i - theta_retrain_i ||^2
    via least squares stacking across trials and coordinates."""
    rows = []
    rhs = []
    for tm, tt, tr in zip(theta_meds, theta_targets, theta_retrains):
        for k in range(tm.shape[0]):
            rows.append([tm[k], tt[k]])
            rhs.append(tr[k])
    A = np.array(rows)
    b = np.array(rhs)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a_opt, b_opt = float(coeffs[0]), float(coeffs[1])
    return a_opt, b_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--summary-out", type=str, default="e2b_summary.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    weights = [1.0, 1.0, 1.0]

    grad_residuals = []
    optimal_residuals = []
    theta_meds, theta_targets, theta_retrains = [], [], []

    for _ in range(args.n_trials):
        clients = setup_trial(rng, n_features=args.n_features,
                              n_samples=args.n_samples)
        theta_med, theta_target, theta_retrain = trial_artifacts(clients,
                                                                 target_idx=0)
        theta_meds.append(theta_med)
        theta_targets.append(theta_target)
        theta_retrains.append(theta_retrain)

        theta_grad = gradient_subtraction(theta_med, theta_target, weights,
                                          target_idx=0)
        grad_residuals.append(np.linalg.norm(theta_grad - theta_retrain))

    # Fit best data-independent linear (a, b) and compute its residual per trial
    a_opt, b_opt = fit_optimal_linear(theta_meds, theta_targets, theta_retrains)
    for tm, tt, tr in zip(theta_meds, theta_targets, theta_retrains):
        theta_opt = a_opt * tm + b_opt * tt
        optimal_residuals.append(float(np.linalg.norm(theta_opt - tr)))

    grad_residuals = np.array(grad_residuals)
    optimal_residuals = np.array(optimal_residuals)

    print("=" * 72)
    print("E.2b  --  Median Aggregator Counterexample")
    print("=" * 72)
    print(f"  Trials                : {args.n_trials}")
    print(f"  Clients per trial     : 3 (target = c_0)")
    print(f"  Features              : {args.n_features}")
    print(f"  Samples per client    : {args.n_samples}")
    print(f"  Aggregator            : coordinate-wise MEDIAN")
    print(f"  Local update          : closed-form ridge "
          "(history-independent)")
    print()

    print("Operator 1 -- Gradient-subtraction analog  (a, b) = (3/2, -1/2):")
    print(f"  mean residual    : {grad_residuals.mean():.3e}")
    print(f"  median residual  : {np.median(grad_residuals):.3e}")
    print(f"  max residual     : {grad_residuals.max():.3e}")
    print(f"  min residual     : {grad_residuals.min():.3e}")
    print()

    print(f"Operator 2 -- OLS-optimal linear  (a, b) = "
          f"({a_opt:.4f}, {b_opt:.4f}):")
    print(f"  mean residual    : {optimal_residuals.mean():.3e}")
    print(f"  median residual  : {np.median(optimal_residuals):.3e}")
    print(f"  max residual     : {optimal_residuals.max():.3e}")
    print(f"  min residual     : {optimal_residuals.min():.3e}")
    print()
    print("=" * 72)

    improvement = grad_residuals.mean() / max(optimal_residuals.mean(), 1e-300)
    print(f"  Mean improvement from optimal vs gradient: {improvement:.2f}x")
    print(f"  But even OLS-optimal has non-zero residual:")
    print(f"  optimal mean = {optimal_residuals.mean():.3e}  -- this is the")
    print(f"  information-loss floor that the characterization theorem")
    print(f"  predicts: no linear Lambda(theta_med, theta_target) can be")
    print(f"  exact for the non-linear median aggregator.")
    print()
    print("  Reading: corroborates the necessity-of-linearity direction of")
    print("  Theorem 2 (report/theorems_draft.tex Sec. 3).")

    summary = {
        "config": {
            "seed": args.seed,
            "n_trials": args.n_trials,
            "n_features": args.n_features,
            "n_samples": args.n_samples,
            "n_clients": 3,
            "target_idx": 0,
        },
        "gradient_subtraction_analog": {
            "weights_a_b": [1.5, -0.5],
            "residual_mean": float(grad_residuals.mean()),
            "residual_median": float(np.median(grad_residuals)),
            "residual_max": float(grad_residuals.max()),
            "residual_min": float(grad_residuals.min()),
        },
        "ols_optimal_linear": {
            "weights_a_b": [a_opt, b_opt],
            "residual_mean": float(optimal_residuals.mean()),
            "residual_median": float(np.median(optimal_residuals)),
            "residual_max": float(optimal_residuals.max()),
            "residual_min": float(optimal_residuals.min()),
        },
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to {args.summary_out}")


if __name__ == "__main__":
    main()
