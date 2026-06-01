/-
# Correctness Theorem (L.2)

The algebraic core of `theorems_draft.tex` Theorem 1: gradient-subtraction
recovers the retain client's local update from the FedAvg aggregate and the
target client's local update.

This module proves the *algebraic identity* underlying the correctness
theorem. The full multi-round correctness statement (Theorem 1 in the
draft) additionally requires a causal-history-independence hypothesis to
lift the single-round identity inductively; that lift is the L.2 follow-on
and depends on a `Trajectory`-level definition that we defer until needed
for Theorem 2 (characterization).
-/

import Theory.FL

namespace FLUL

/-- **Theorem 1 (algebraic core, 2 clients)**: given the FedAvg aggregate
of two clients with positive weights and the target client's local update,
gradient-subtraction recovers the retain client's local update *exactly*.

This is a pure algebraic identity over `ℝ`; no history-independence is
required for the single-round case. Empirically corroborated by E.1 test 1
(`run_e1_linear_regression.py`, residual `3.25 × 10⁻¹⁶`). -/
theorem gradUnlearn_fedAvg2
    (θ_target θ_retain w_target w_retain : ℝ)
    (hw_retain : w_retain ≠ 0)
    (hW : w_target + w_retain ≠ 0) :
    gradUnlearn
      ((w_target * θ_target + w_retain * θ_retain) / (w_target + w_retain))
      θ_target
      w_target
      (w_target + w_retain)
    = θ_retain := by
  unfold gradUnlearn
  have h_diff : (w_target + w_retain) - w_target = w_retain := by ring
  rw [h_diff]
  field_simp
  ring

/-- **Theorem 1 (symmetric case)**: gradient-subtraction at `w_target = 1`,
`W = 2` (equal weights, 2 clients) — the form used in `theorems_draft.tex`
Theorem 3a. Reduces to `2 θ̄ − θ_★ = θ_r` where `θ̄ = (θ_★ + θ_r) / 2`. -/
theorem gradUnlearn_fedAvg2_symmetric (θ_target θ_retain : ℝ) :
    gradUnlearn ((θ_target + θ_retain) / 2) θ_target 1 2 = θ_retain := by
  -- Specialize gradUnlearn_fedAvg2 to w_target = w_retain = 1
  have h1 : (1 : ℝ) * θ_target + 1 * θ_retain = θ_target + θ_retain := by ring
  have h2 : (1 : ℝ) + 1 = 2 := by norm_num
  have := gradUnlearn_fedAvg2 θ_target θ_retain 1 1
            (by norm_num) (by norm_num)
  rw [h1, h2] at this
  exact this

end FLUL
