/-
# Multi-dim Theorem 3a (L.4-MD)

Polymorphic generalization of `fedProx2_gradUnlearn_residual` (the scalar
μ-coupling residual under 2-client, T=2 FedProx) to arbitrary finite
dimension `d`.

Setup. Each client `c ∈ {target, retain}` holds a Gram matrix
`α_c : Matrix d d ℝ` (`α_c = X_c^T X_c / n_c` in the ridge-regression
interpretation) and a data vector `β_c : d → ℝ`. With FedProx proximal
coefficient `μ`, equal client weights, FedAvg aggregator, the closed-form
FedProx local solver is
    `θ_c^(t) = (α_c + μ I)⁻¹ · (β_c + μ · θ^(t-1))`,
`θ^(0) = 0`.

Two theorems:

* `fedProx2_gradUnlearn_residual_md`: at round 2,
    `(2 · θ² − θ_★²) − θ_retrain² = (μ/2) · (α_r + μ I)⁻¹ · (θ_★¹ − θ_r¹)`,
  where `θ_★¹ - θ_r¹` is the round-1 client disagreement. The identity is
  purely formal — uses only linearity of `Matrix.mulVec` and holds for any
  matrices `α_t, α_r` (the junk value of `M⁻¹` for singular `M` cancels).

* `fedProx2_gradUnlearn_residual_md_nonzero`: under `Invertible (α_r + μ•1)`,
  `μ > 0`, and a round-1 disagreement `θ_★¹ ≠ θ_r¹`, the residual is
  strictly nonzero. This mirrors `fedProx2_gradUnlearn_residual_nonzero`
  in `Theory.Impossibility` (the scalar version).

The `Invertible (α_r + μ•1)` hypothesis is automatic under `α_r.PosSemidef`
and `μ > 0` (positive-definite Gram matrix). A bridge lemma deriving it
from PSD is left as future work; users with `PosSemidef` data can call
`Matrix.PosDef.invertible` after establishing `(α_r + μ•1).PosDef`
themselves.

Specializing the residual identity to `d = Fin 1` recovers the scalar
Theorem 3a from `Impossibility.lean`.

Empirical corroboration: E.3 (`run_e3_impossibility_toy.py`) verifies the
scalar `d = 1` case. A vector-valued E.3 variant would corroborate the
multi-dim form directly; not on the current critical path.
-/

import Theory.FL
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Analysis.RCLike.Basic

namespace FLUL

section MultiDim

variable {d : Type*} [Fintype d] [DecidableEq d]

/-- **Multi-dim Theorem 3a**: closed-form gradient-subtraction residual for
2-client, T=2 FedProx with equal weights, in arbitrary finite dimension `d`.

The matrix `M_c := α_c + μ I` is the regularized Gram matrix per client.
With `M_c⁻¹` (junk-valued if `M_c` is singular) the FedProx solver is
defined for all data. The residual identity below uses only linearity of
`Matrix.mulVec` and therefore holds without invertibility hypotheses; it
is meaningful (a true linear-algebra identity) when `M_r` is invertible,
which is automatic under `α_r.PosSemidef` and `μ > 0`.

Empirical corroboration via scalar specialization: E.3 verifies the `d = 1`
case to `~10⁻¹⁶` over 1 000 trials per μ. -/
theorem fedProx2_gradUnlearn_residual_md
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ) :
    let M_t : Matrix d d ℝ := α_t + μ • (1 : Matrix d d ℝ)
    let M_r : Matrix d d ℝ := α_r + μ • (1 : Matrix d d ℝ)
    let θ_t_1 := M_t⁻¹.mulVec β_t
    let θ_r_1 := M_r⁻¹.mulVec β_r
    let θ_1 := (1/2 : ℝ) • (θ_t_1 + θ_r_1)
    let θ_t_2 := M_t⁻¹.mulVec (β_t + μ • θ_1)
    let θ_r_2 := M_r⁻¹.mulVec (β_r + μ • θ_1)
    let θ_2 := (1/2 : ℝ) • (θ_t_2 + θ_r_2)
    let θ_retrain_2 := M_r⁻¹.mulVec (β_r + μ • θ_r_1)
    ((2 : ℝ) • θ_2 - θ_t_2) - θ_retrain_2 =
      (μ / 2) • M_r⁻¹.mulVec (θ_t_1 - θ_r_1) := by
  intro M_t M_r θ_t_1 θ_r_1 θ_1 θ_t_2 θ_r_2 θ_2 θ_retrain_2
  -- Step 1: FedAvg-2 identity (vector form). 2 • θ_2 - θ_t_2 = θ_r_2.
  have h_grad : (2 : ℝ) • θ_2 - θ_t_2 = θ_r_2 := by
    simp only [θ_2]
    ext i
    simp [Pi.sub_apply, Pi.add_apply]
  rw [h_grad]
  -- Step 2: pre-image difference factors out (μ/2) and a vector subtraction.
  have h_diff :
      (β_r + μ • θ_1) - (β_r + μ • θ_r_1) = (μ / 2) • (θ_t_1 - θ_r_1) := by
    simp only [θ_1]
    ext i
    simp [Pi.sub_apply, Pi.add_apply, Pi.smul_apply]
    ring
  -- Step 3: linearity of mulVec carries the algebra through.
  simp only [θ_r_2, θ_retrain_2]
  rw [← Matrix.mulVec_sub, h_diff, Matrix.mulVec_smul]

/-! ### Strengthening: non-vanishing residual

The residual identity above is purely formal — it doesn't say the residual
is nonzero. Under invertibility of the regularized Gram matrix `M_r` and
a round-1 disagreement, the residual is genuinely nonzero. -/

/-- For invertible `M`, the inverse's `mulVec` action has trivial kernel:
`M⁻¹.mulVec v = 0 ↔ v = 0`. -/
private lemma mulVec_inv_eq_zero_iff
    (M : Matrix d d ℝ) [Invertible M] (v : d → ℝ) :
    M⁻¹.mulVec v = 0 ↔ v = 0 := by
  refine ⟨fun h => ?_, fun h => by rw [h]; exact Matrix.mulVec_zero _⟩
  have h1 : M.mulVec (M⁻¹.mulVec v) = 0 := by rw [h, Matrix.mulVec_zero]
  rwa [Matrix.mulVec_mulVec, Matrix.mul_inv_of_invertible, Matrix.one_mulVec] at h1

/-- **Multi-dim Theorem 3a, non-vanishing form**: when the regularized Gram
matrix `α_r + μ • 1` is invertible, `μ > 0`, and the two clients' round-1
closed-form solutions disagree, the gradient-subtraction residual is
strictly nonzero.

This mirrors `fedProx2_gradUnlearn_residual_nonzero` in
`Theory.Impossibility` (scalar version). Under `α_r.PosSemidef` and `μ > 0`,
the `Invertible` hypothesis is automatic (positive-definite matrices are
invertible). -/
theorem fedProx2_gradUnlearn_residual_md_nonzero
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ)
    (hμ : 0 < μ)
    [Invertible (α_r + μ • (1 : Matrix d d ℝ))]
    (hdisagree :
      (α_t + μ • (1 : Matrix d d ℝ))⁻¹.mulVec β_t ≠
      (α_r + μ • (1 : Matrix d d ℝ))⁻¹.mulVec β_r) :
    let M_t : Matrix d d ℝ := α_t + μ • (1 : Matrix d d ℝ)
    let M_r : Matrix d d ℝ := α_r + μ • (1 : Matrix d d ℝ)
    let θ_t_1 := M_t⁻¹.mulVec β_t
    let θ_r_1 := M_r⁻¹.mulVec β_r
    let θ_1 := (1/2 : ℝ) • (θ_t_1 + θ_r_1)
    let θ_t_2 := M_t⁻¹.mulVec (β_t + μ • θ_1)
    let θ_r_2 := M_r⁻¹.mulVec (β_r + μ • θ_1)
    let θ_2 := (1/2 : ℝ) • (θ_t_2 + θ_r_2)
    let θ_retrain_2 := M_r⁻¹.mulVec (β_r + μ • θ_r_1)
    ((2 : ℝ) • θ_2 - θ_t_2) - θ_retrain_2 ≠ 0 := by
  intro M_t M_r θ_t_1 θ_r_1 θ_1 θ_t_2 θ_r_2 θ_2 θ_retrain_2
  rw [fedProx2_gradUnlearn_residual_md α_t α_r β_t β_r μ]
  -- Goal: (μ/2) • M_r⁻¹.mulVec (θ_t_1 - θ_r_1) ≠ 0
  refine smul_ne_zero ?_ ?_
  · -- μ/2 ≠ 0
    exact div_ne_zero (ne_of_gt hμ) two_ne_zero
  · -- M_r⁻¹.mulVec (θ_t_1 - θ_r_1) ≠ 0
    intro h_zero
    rw [mulVec_inv_eq_zero_iff] at h_zero
    exact hdisagree (sub_eq_zero.mp h_zero)

/-! ### PosSemidef bridge

In ridge regression `α_c = X_c^T X_c / n_c ≥ 0` is PSD, so users would
prefer `α_r.PosSemidef + μ > 0` as the hypothesis instead of `Invertible`.
This block builds that bridge:

1. `μ • I_d` is positive definite when `μ > 0`
   (`smul_one_posDef`, via `Matrix.PosDef.diagonal`).
2. PSD + PosDef ⇒ PosDef (`posSemidef_add_smul_one_posDef`, via
   `Matrix.PosDef.posSemidef_add`).
3. PosDef ⇒ `Invertible` (`invertibleOfPosSemidefAddSmulOne`, via
   `Matrix.PosDef.isUnit` and `Matrix.isUnit_iff_isUnit_det` ↣
   `Matrix.invertibleOfIsUnitDet`).
4. PSD-based corollary `fedProx2_gradUnlearn_residual_md_nonzero_of_posSemidef`
   restates the non-vanishing theorem under the natural ridge-regression
   hypothesis. -/

section PSDBridge

variable {d' : Type*} [DecidableEq d']

/-- `μ • I_{d'} = diagonal (const μ)` — entrywise identity. -/
private lemma smul_one_eq_diagonal_const (μ : ℝ) :
    (μ • (1 : Matrix d' d' ℝ)) = Matrix.diagonal (fun _ : d' => μ) := by
  ext i j
  by_cases h : i = j
  · subst h
    simp [Matrix.smul_apply, Matrix.one_apply_eq, Matrix.diagonal_apply_eq]
  · simp [Matrix.smul_apply, Matrix.one_apply_ne h, Matrix.diagonal_apply_ne _ h]

/-- For `0 < μ`, the scalar multiple `μ • I_{d'}` is positive definite.

`Matrix.PosDef.diagonal` needs `[StarOrderedRing R]`. Mathlib registers
this as a lemma `RCLike.toStarOrderedRing` rather than a global instance
(to avoid typeclass diamonds elsewhere), so we install it locally. -/
lemma smul_one_posDef (μ : ℝ) (hμ : 0 < μ) :
    (μ • (1 : Matrix d' d' ℝ)).PosDef := by
  haveI : StarOrderedRing ℝ := RCLike.toStarOrderedRing
  rw [smul_one_eq_diagonal_const μ]
  exact Matrix.PosDef.diagonal (fun _ => hμ)

/-- **PSD bridge — PosDef sum**: for `α : Matrix d' d' ℝ` positive
semidefinite and `μ > 0`, the regularised Gram matrix `α + μ • I` is
positive definite. -/
lemma posSemidef_add_smul_one_posDef
    {α : Matrix d' d' ℝ} (hα : α.PosSemidef) {μ : ℝ} (hμ : 0 < μ) :
    (α + μ • (1 : Matrix d' d' ℝ)).PosDef :=
  Matrix.PosDef.posSemidef_add hα (smul_one_posDef μ hμ)

end PSDBridge

/-- **PSD bridge — Invertible**: `α.PosSemidef` + `μ > 0` ⇒
`Invertible (α + μ • I)`. The non-vanishing theorem
`fedProx2_gradUnlearn_residual_md_nonzero` takes `Invertible` as a
typeclass hypothesis; this lemma constructs that instance from the
natural ridge-regression hypothesis. -/
@[reducible]
noncomputable def invertibleOfPosSemidefAddSmulOne
    {α : Matrix d d ℝ} (hα : α.PosSemidef) {μ : ℝ} (hμ : 0 < μ) :
    Invertible (α + μ • (1 : Matrix d d ℝ)) :=
  have hIsUnit : IsUnit ((α + μ • (1 : Matrix d d ℝ)).det) := by
    rw [← Matrix.isUnit_iff_isUnit_det]
    exact (posSemidef_add_smul_one_posDef hα hμ).isUnit
  Matrix.invertibleOfIsUnitDet _ hIsUnit

/-- **Multi-dim Theorem 3a, non-vanishing form under PSD**: the natural
ridge-regression restatement. Under `α_r.PosSemidef`, `μ > 0`, and a
round-1 disagreement, the gradient-subtraction residual is strictly
nonzero — no explicit `Invertible` instance required at the call site. -/
theorem fedProx2_gradUnlearn_residual_md_nonzero_of_posSemidef
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ)
    (hμ : 0 < μ)
    (hα_r : α_r.PosSemidef)
    (hdisagree :
      (α_t + μ • (1 : Matrix d d ℝ))⁻¹.mulVec β_t ≠
      (α_r + μ • (1 : Matrix d d ℝ))⁻¹.mulVec β_r) :
    let M_t : Matrix d d ℝ := α_t + μ • (1 : Matrix d d ℝ)
    let M_r : Matrix d d ℝ := α_r + μ • (1 : Matrix d d ℝ)
    let θ_t_1 := M_t⁻¹.mulVec β_t
    let θ_r_1 := M_r⁻¹.mulVec β_r
    let θ_1 := (1/2 : ℝ) • (θ_t_1 + θ_r_1)
    let θ_t_2 := M_t⁻¹.mulVec (β_t + μ • θ_1)
    let θ_r_2 := M_r⁻¹.mulVec (β_r + μ • θ_1)
    let θ_2 := (1/2 : ℝ) • (θ_t_2 + θ_r_2)
    let θ_retrain_2 := M_r⁻¹.mulVec (β_r + μ • θ_r_1)
    ((2 : ℝ) • θ_2 - θ_t_2) - θ_retrain_2 ≠ 0 := by
  haveI : Invertible (α_r + μ • (1 : Matrix d d ℝ)) :=
    invertibleOfPosSemidefAddSmulOne hα_r hμ
  exact fedProx2_gradUnlearn_residual_md_nonzero α_t α_r β_t β_r μ hμ hdisagree

end MultiDim

end FLUL
