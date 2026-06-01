"""
E.3 -- 2-Client, 2-Round Impossibility Toy
============================================

Empirical corroboration of Theorem 3
(report/theorems_draft.tex Sec. 4):

    For 2-client, 2-round scalar FedProx with mu > 0, no data-independent
    linear unlearning operator
        Lambda(theta^{(2)}, theta_target^{(2)}) = a theta^{(2)} + b theta_target^{(2)}
    can recover theta_retrain^{(2)} exactly. The gradient-subtraction analog
    (a, b) = (W / (W - w), - w / (W - w)) = (2, -1) for w = 1, W = 2 has
    residual delta_grad(mu) = (mu / (2(alpha_2 + mu))) * (theta_target^{(1)}
    - theta_retain^{(1)}), and even the OLS-optimal data-independent (a, b)
    is bounded below away from zero whenever the data distribution has
    sufficient variability.

We compare three quantities across mu:

  1. Analytical gradient-subtraction residual
        delta_grad(mu) = (mu / (2(alpha_2 + mu))) * (theta_1^{(1)} - theta_2^{(1)})
     computed in closed form per trial.
  2. Numerical gradient-subtraction residual: the same quantity, but computed
     by running the protocol step-by-step. Sanity check on the closed form.
  3. OLS-optimal linear operator residual: the minimum data-independent
     (a, b) residual over n_trials data instances.

Predictions:
  - mu = 0   : delta_grad = 0 exactly. Numerical agrees. OLS-optimal achieves
                zero residual since (a, b) = (2, -1) is exact for all data.
  - mu > 0   : delta_grad > 0. Numerical agrees with the closed form to
                machine precision. OLS-optimal beats gradient subtraction
                but is itself bounded below away from zero.

Run:
    python run_e3_impossibility_toy.py [--seed 42] [--n-trials 1000]
"""

import argparse
import json
import numpy as np


def setup_trial(rng, n_samples=50, true_w_std=2.0, noise_std=0.1):
    """One scalar federated linear-regression trial: two clients, each with
    their own random ground-truth coefficient (so their local closed-form
    solutions disagree)."""
    X1 = rng.standard_normal((n_samples, 1))
    w1 = true_w_std * rng.standard_normal()
    y1 = X1[:, 0] * w1 + noise_std * rng.standard_normal(n_samples)
    X2 = rng.standard_normal((n_samples, 1))
    w2 = true_w_std * rng.standard_normal()
    y2 = X2[:, 0] * w2 + noise_std * rng.standard_normal(n_samples)
    return X1, y1, X2, y2


def scalar_alpha_beta(X, y):
    """For scalar (d=1) regression: alpha = X^T X / n, beta = X^T y / n."""
    n = X.shape[0]
    alpha = float((X.T @ X)[0, 0]) / n
    beta = float((X.T @ y)[0]) / n
    return alpha, beta


def run_protocol(alphas, betas, mu, n_rounds, weights):
    """Run scalar FedProx for n_rounds with closed-form local minimization.
    Returns the trajectory of aggregates and the final round's client thetas."""
    W = sum(weights)
    theta = 0.0
    last_client_thetas = None
    for _ in range(n_rounds):
        client_thetas = [(b + mu * theta) / (a + mu)
                         for a, b in zip(alphas, betas)]
        theta = sum(w * t for w, t in zip(weights, client_thetas)) / W
        last_client_thetas = client_thetas
    return theta, last_client_thetas


def analytical_delta_grad(alpha_target, beta_target, alpha_retain, beta_retain,
                          mu, T=2):
    """Closed-form gradient-subtraction residual at round T for the 2-client,
    equal-weight symmetric case.

    For T = 2 this evaluates to
        delta = (mu / (2(alpha_retain + mu))) * (theta_target^{(1)}
                                                  - theta_retain^{(1)})
    where theta_c^{(1)} = beta_c / (alpha_c + mu).
    """
    if T != 2:
        raise NotImplementedError("Closed form only derived for T = 2.")
    theta_target_1 = beta_target / (alpha_target + mu)
    theta_retain_1 = beta_retain / (alpha_retain + mu)
    return (mu / (2 * (alpha_retain + mu))) * (theta_target_1 - theta_retain_1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument(
        "--mu-list", type=str, default="0,0.001,0.01,0.1,1,10",
        help="Comma-separated proximal coefficients to sweep.")
    parser.add_argument("--summary-out", type=str, default="e3_summary.json")
    args = parser.parse_args()

    mu_list = [float(s) for s in args.mu_list.split(",")]
    rng = np.random.default_rng(args.seed)
    n_rounds = 2  # fixed by the theorem's setup
    weights = [1.0, 1.0]

    trials = [setup_trial(rng, n_samples=args.n_samples)
              for _ in range(args.n_trials)]
    alpha_betas = [
        (scalar_alpha_beta(X1, y1), scalar_alpha_beta(X2, y2))
        for (X1, y1, X2, y2) in trials
    ]

    print("=" * 72)
    print("E.3  --  2-Client, 2-Round Impossibility Toy")
    print("=" * 72)
    print(f"  Trials                : {args.n_trials}")
    print(f"  Clients per trial     : 2 (target = c_0)")
    print(f"  Rounds                : 2")
    print(f"  Scalar regression     : d = 1, n_samples = {args.n_samples}")
    print(f"  Aggregator            : FedAvg, equal weights")
    print(f"  Local solver          : closed-form FedProx")
    print(f"  Seed                  : {args.seed}")
    print()

    print(f"  {'mu':>10s}  {'analytical':>14s}  {'numerical':>14s}  "
          f"{'opt linear':>14s}  {'opt (a, b)':>22s}")
    print("  " + "-" * 80)

    rows = []
    for mu in mu_list:
        # Numerical and analytical gradient-subtraction residuals per trial
        analytical_resids = []
        numerical_resids = []
        theta_aggs = []
        theta_targets = []
        theta_retrains = []
        for (ab1, ab2) in alpha_betas:
            a1, b1 = ab1
            a2, b2 = ab2
            alphas = [a1, a2]
            betas = [b1, b2]

            # Original protocol
            theta_agg_2, last_orig = run_protocol(alphas, betas, mu, n_rounds, weights)
            theta_target_2 = last_orig[0]

            # True retrain: only client 1 (= c_2 in math), with the same protocol
            theta_retrain_2, _ = run_protocol([a2], [b2], mu, n_rounds, [1.0])

            # Numerical gradient subtraction
            theta_grad = 2 * theta_agg_2 - theta_target_2
            numerical_resids.append(abs(theta_grad - theta_retrain_2))

            # Analytical gradient subtraction residual
            ad = analytical_delta_grad(a1, b1, a2, b2, mu, T=n_rounds)
            analytical_resids.append(abs(ad))

            theta_aggs.append(theta_agg_2)
            theta_targets.append(theta_target_2)
            theta_retrains.append(theta_retrain_2)

        # OLS-optimal data-independent linear (a, b)
        A = np.column_stack([theta_aggs, theta_targets])
        b = np.array(theta_retrains)
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        a_opt, b_opt = float(coeffs[0]), float(coeffs[1])
        opt_preds = A @ coeffs
        opt_resids = np.abs(opt_preds - b)

        ana_mean = float(np.mean(analytical_resids))
        num_mean = float(np.mean(numerical_resids))
        opt_mean = float(opt_resids.mean())

        print(f"  {mu:>10.4g}  {ana_mean:>14.3e}  {num_mean:>14.3e}  "
              f"{opt_mean:>14.3e}  ({a_opt:>+.4f}, {b_opt:>+.4f})")

        rows.append({
            "mu": mu,
            "analytical_grad_residual_mean": ana_mean,
            "numerical_grad_residual_mean": num_mean,
            "optimal_linear_residual_mean": opt_mean,
            "optimal_linear_a": a_opt,
            "optimal_linear_b": b_opt,
        })

    print()
    print("=" * 72)

    # Numerical / analytical agreement check at mu > 0
    diffs = [abs(r["analytical_grad_residual_mean"] - r["numerical_grad_residual_mean"])
             for r in rows]
    max_diff = max(diffs)
    print(f"  Closed-form vs numerical max |diff|: {max_diff:.3e}")
    if max_diff < 1e-12:
        print("  -> Analytical and numerical gradient residuals agree to "
              "machine precision.")
    print()
    print("  Reading: gradient-subtraction's residual grows with mu in the small")
    print("  regime (linear scaling) and saturates at large mu. The OLS-optimal")
    print("  linear operator is bounded below away from zero for every mu > 0 --")
    print("  the empirical face of the impossibility theorem.")
    print()
    print("  Connection to E.2: E.2 measured the residual in a multi-feature")
    print("  vector setting at T = 20 rounds. E.3 isolates the same effect in")
    print("  the minimal 2-client, 2-round, d = 1 setting where the residual")
    print("  has a closed form.")

    summary = {
        "config": {
            "seed": args.seed,
            "n_trials": args.n_trials,
            "n_samples": args.n_samples,
            "n_clients": 2,
            "n_rounds": n_rounds,
            "weights": weights,
            "mu_list": mu_list,
        },
        "results": rows,
        "max_analytical_numerical_diff": max_diff,
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to {args.summary_out}")


if __name__ == "__main__":
    main()
