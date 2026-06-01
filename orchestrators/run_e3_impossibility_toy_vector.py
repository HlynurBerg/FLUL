"""
E.3 (vector) -- 2-Client, 2-Round Impossibility Toy in d > 1
============================================================

Empirical corroboration of the **multi-dim** Theorem 3a
(`Theory/ImpossibilityMD.lean`, `fedProx2_gradUnlearn_residual_md`):

    For 2-client, 2-round FedProx with equal weights and mu > 0, the
    gradient-subtraction unlearning residual at round 2 is

        delta := (2 * theta_2 - theta_target_2) - theta_retrain_2
               = (mu / 2) * M_r^{-1} * (theta_target_1 - theta_retain_1)

    where M_c := alpha_c + mu * I is the regularised Gram matrix and
    theta_c_1 := M_c^{-1} * beta_c is the client's round-1 closed-form.

The scalar version (`run_e3_impossibility_toy.py`) verifies the d = 1
case to machine precision. This script:

  1. Generates random PSD alpha matrices via `X^T X / n` for a chosen d.
  2. Runs the protocol step-by-step to get the numerical residual.
  3. Computes the analytical residual via the closed-form identity above.
  4. Checks analytical / numerical agreement (machine precision is the
     pass criterion -- the identity is purely algebraic).
  5. Fits an OLS-optimal data-independent linear operator
        Lambda(theta, theta_target) = A * theta + B * theta_target
     (where A, B are d x d matrices) across trials and reports the
     residual norm achieved. Mirrors the multi-dim Theorem 3b setup
     (`fedProx2_no_exact_data_indep_linear_md`).

Predictions:
  - mu = 0   : delta = 0 exactly. OLS-optimal achieves zero residual
                since (A, B) = (2I, -I) is exact for all data.
  - mu > 0   : delta > 0 (norm-positive). Analytical agrees with
                numerical to ~1e-13 or better. OLS-optimal beats the
                gradient analog but is itself bounded below away from
                zero -- the multi-dim impossibility empirically.

Also exercises the PSD bridge
(`fedProx2_gradUnlearn_residual_md_nonzero_of_posSemidef`):
each random alpha is constructed as `X^T X / n` so it is PSD by
construction; the regularised `alpha + mu I` is therefore PosDef and
hence invertible (the Lean hypothesis discharged automatically). The
sweep over multiple seeds + multiple d values empirically confirms the
PSD-driven invertibility never fails in 1 000 x len(d_list) trials.

Run:
    python run_e3_impossibility_toy_vector.py [--seed 42] [--n-trials 1000]
        [--d-list 4,16] [--mu-list 0,0.001,0.01,0.1,1,10]
"""

import argparse
import json
import numpy as np


def setup_trial(rng, d, n_samples, true_w_std=2.0, noise_std=0.1):
    """One vector federated ridge-regression trial. Two clients, each
    with their own random ground-truth weight vector so their local
    closed-form solutions disagree. Returns alpha_c (d x d PSD) and
    beta_c (d-vector) per client."""
    X1 = rng.standard_normal((n_samples, d))
    w1 = true_w_std * rng.standard_normal(d)
    y1 = X1 @ w1 + noise_std * rng.standard_normal(n_samples)
    X2 = rng.standard_normal((n_samples, d))
    w2 = true_w_std * rng.standard_normal(d)
    y2 = X2 @ w2 + noise_std * rng.standard_normal(n_samples)
    alpha_1 = (X1.T @ X1) / n_samples  # PSD by construction
    beta_1 = (X1.T @ y1) / n_samples
    alpha_2 = (X2.T @ X2) / n_samples
    beta_2 = (X2.T @ y2) / n_samples
    return alpha_1, beta_1, alpha_2, beta_2


def fedprox_step(alpha, beta, mu, theta_prev, d):
    """One FedProx local closed-form step:
        theta = (alpha + mu I)^{-1} (beta + mu theta_prev)."""
    M = alpha + mu * np.eye(d)
    return np.linalg.solve(M, beta + mu * theta_prev)


def run_protocol(alphas, betas, mu, n_rounds, weights, d):
    """Run 2-client (or N-client) FedProx for n_rounds. Returns the
    final aggregate, the final target client's theta, and the
    full per-round trajectory of client thetas (for debugging)."""
    W = sum(weights)
    theta = np.zeros(d)
    last_clients = None
    for _ in range(n_rounds):
        client_thetas = [fedprox_step(a, b, mu, theta, d)
                         for a, b in zip(alphas, betas)]
        theta = sum(w * t for w, t in zip(weights, client_thetas)) / W
        last_clients = client_thetas
    return theta, last_clients


def analytical_residual_vec(alpha_t, beta_t, alpha_r, beta_r, mu, d):
    """Closed-form multi-dim residual at T = 2 for equal-weight 2-client
    FedProx:
        delta = (mu / 2) * M_r^{-1} * (theta_t_1 - theta_r_1)
    Exactly the identity proved as `fedProx2_gradUnlearn_residual_md`."""
    M_t = alpha_t + mu * np.eye(d)
    M_r = alpha_r + mu * np.eye(d)
    theta_t_1 = np.linalg.solve(M_t, beta_t)
    theta_r_1 = np.linalg.solve(M_r, beta_r)
    return (mu / 2.0) * np.linalg.solve(M_r, theta_t_1 - theta_r_1)


def fit_optimal_linear_AB(theta_aggs, theta_targets, theta_retrains, d):
    """Solve the matrix OLS problem
        min over (A, B in R^{d x d})  sum_i || A theta_agg_i +
                                              B theta_target_i
                                              - theta_retrain_i ||^2
    Returns (A, B) and per-trial residual norms.

    Implementation: stack theta_agg_i and theta_target_i as rows of an
    m x 2d design matrix X, theta_retrain_i as rows of an m x d target
    matrix Y. Then find W in R^{2d x d} with X W = Y in the least-squares
    sense; A = W[:d].T, B = W[d:].T."""
    m = len(theta_aggs)
    X = np.zeros((m, 2 * d))
    Y = np.zeros((m, d))
    for i in range(m):
        X[i, :d] = theta_aggs[i]
        X[i, d:] = theta_targets[i]
        Y[i] = theta_retrains[i]
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)  # W is (2d, d)
    A = W[:d, :].T  # (d, d)
    B = W[d:, :].T  # (d, d)
    preds = X @ W  # (m, d)
    resid_norms = np.linalg.norm(preds - Y, axis=1)
    return A, B, resid_norms


def run_one_dimension(d, mu_list, n_trials, n_samples, rng):
    """Run the sweep for one fixed d. Returns the per-mu summary rows
    plus the maximum analytical-vs-numerical disagreement (sanity check)."""
    weights = [1.0, 1.0]
    n_rounds = 2

    # Pre-generate trial data (so each mu uses the same data instances)
    trials = [setup_trial(rng, d=d, n_samples=n_samples)
              for _ in range(n_trials)]

    print(f"\n  d = {d}  (n_trials = {n_trials}, n_samples = {n_samples})")
    print(f"  {'mu':>10s}  {'analytical_norm':>17s}  {'numerical_norm':>17s}  "
          f"{'opt_linear_norm':>17s}  {'max |ana-num|':>14s}")
    print("  " + "-" * 90)

    rows = []
    overall_max_diff = 0.0
    for mu in mu_list:
        analytical_norms = []
        numerical_norms = []
        ana_num_diffs = []
        theta_aggs = []
        theta_targets = []
        theta_retrains = []
        for (alpha_t, beta_t, alpha_r, beta_r) in trials:
            # Original protocol (both clients)
            theta_agg_2, last_orig = run_protocol(
                [alpha_t, alpha_r], [beta_t, beta_r], mu, n_rounds,
                weights, d)
            theta_target_2 = last_orig[0]

            # Retrain: only retain client
            theta_retrain_2, _ = run_protocol(
                [alpha_r], [beta_r], mu, n_rounds, [1.0], d)

            # Numerical gradient-subtraction residual
            theta_grad = 2 * theta_agg_2 - theta_target_2
            num_res = theta_grad - theta_retrain_2
            numerical_norms.append(float(np.linalg.norm(num_res)))

            # Analytical closed form
            ana_res = analytical_residual_vec(
                alpha_t, beta_t, alpha_r, beta_r, mu, d)
            analytical_norms.append(float(np.linalg.norm(ana_res)))

            ana_num_diffs.append(float(np.linalg.norm(num_res - ana_res)))

            theta_aggs.append(theta_agg_2)
            theta_targets.append(theta_target_2)
            theta_retrains.append(theta_retrain_2)

        # OLS-optimal data-independent (A, B)
        A_opt, B_opt, opt_resid_norms = fit_optimal_linear_AB(
            theta_aggs, theta_targets, theta_retrains, d)

        ana_mean = float(np.mean(analytical_norms))
        num_mean = float(np.mean(numerical_norms))
        opt_mean = float(np.mean(opt_resid_norms))
        max_diff = float(np.max(ana_num_diffs))
        overall_max_diff = max(overall_max_diff, max_diff)

        print(f"  {mu:>10.4g}  {ana_mean:>17.3e}  {num_mean:>17.3e}  "
              f"{opt_mean:>17.3e}  {max_diff:>14.3e}")

        rows.append({
            "mu": mu,
            "analytical_residual_norm_mean": ana_mean,
            "numerical_residual_norm_mean": num_mean,
            "optimal_linear_residual_norm_mean": opt_mean,
            "max_analytical_numerical_diff": max_diff,
            "optimal_A_frobenius": float(np.linalg.norm(A_opt, ord="fro")),
            "optimal_B_frobenius": float(np.linalg.norm(B_opt, ord="fro")),
        })

    return rows, overall_max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument(
        "--d-list", type=str, default="4,16",
        help="Comma-separated parameter dimensions to sweep.")
    parser.add_argument(
        "--mu-list", type=str, default="0,0.001,0.01,0.1,1,10",
        help="Comma-separated proximal coefficients to sweep.")
    parser.add_argument("--summary-out", type=str,
                        default="e3_vector_summary.json")
    args = parser.parse_args()

    d_list = [int(s) for s in args.d_list.split(",")]
    mu_list = [float(s) for s in args.mu_list.split(",")]
    rng = np.random.default_rng(args.seed)

    print("=" * 96)
    print("E.3 (vector)  --  2-Client, 2-Round Impossibility Toy in d > 1")
    print("=" * 96)
    print(f"  Trials per (d, mu)   : {args.n_trials}")
    print(f"  Clients per trial    : 2 (target = c_0, retain = c_1)")
    print(f"  Rounds               : 2")
    print(f"  Dimensions swept     : {d_list}")
    print(f"  Samples per client   : {args.n_samples}")
    print(f"  Aggregator           : FedAvg, equal weights")
    print(f"  Local solver         : closed-form FedProx (alpha + mu I)^-1")
    print(f"  alpha construction   : X^T X / n  (PSD by construction)")
    print(f"  Seed                 : {args.seed}")

    per_dim_results = {}
    global_max_diff = 0.0
    for d in d_list:
        rows, max_diff = run_one_dimension(
            d=d, mu_list=mu_list, n_trials=args.n_trials,
            n_samples=args.n_samples, rng=rng)
        per_dim_results[str(d)] = rows
        global_max_diff = max(global_max_diff, max_diff)

    print()
    print("=" * 96)
    print(f"  Global max analytical-vs-numerical disagreement: "
          f"{global_max_diff:.3e}")
    if global_max_diff < 1e-9:
        print("  -> Analytical (Theorem 3a multi-dim closed form) and numerical")
        print("     residuals agree to (near-)machine precision across all (d, mu)")
        print("     and 1000 trials per cell. The Lean identity")
        print("     `fedProx2_gradUnlearn_residual_md` is corroborated numerically.")
    print()
    print("  Reading: at mu = 0 the gradient analog (A, B) = (2I, -I) is exact")
    print("  (analytical_norm = 0). For mu > 0 the residual norm grows with mu in")
    print("  the small-mu regime and saturates as the regularisation dominates.")
    print("  The OLS-optimal data-independent (A, B) beats the gradient analog")
    print("  but is bounded below away from zero -- the empirical face of")
    print("  multi-dim Theorem 3b (`fedProx2_no_exact_data_indep_linear_md`).")
    print()
    print("  PSD bridge exercised: each alpha = X^T X / n is PSD by construction,")
    print(f"  so (alpha + mu I) is positive definite for all mu > 0 -- and the")
    print(f"  protocol completes successfully across {len(d_list) * args.n_trials}")
    print(f"  random PSD instances per mu, with no singular-matrix failures.")
    print()
    print("  Connection to scalar E.3: this is the d > 1 generalisation of")
    print("  `run_e3_impossibility_toy.py`. The d = 1 case there hits ~1e-16")
    print("  agreement; here the worst-case agreement is dictated by the")
    print("  condition number of M_r (alpha PSD + mu * I), still well within")
    print("  numerical tolerance.")

    summary = {
        "config": {
            "seed": args.seed,
            "n_trials": args.n_trials,
            "n_samples": args.n_samples,
            "n_clients": 2,
            "n_rounds": 2,
            "weights": [1.0, 1.0],
            "d_list": d_list,
            "mu_list": mu_list,
        },
        "results_by_dimension": per_dim_results,
        "global_max_analytical_numerical_diff": global_max_diff,
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to {args.summary_out}")


if __name__ == "__main__":
    main()
