/-
# Multi-dim Theorem 3b (L.4-MDB)

Vector generalization of `fedProx2_no_exact_data_indep_linear` from
`Theory.Impossibility`. We strengthen 3b to allow *matrix* coefficients
`(A, B) ∈ Matrix d d ℝ × Matrix d d ℝ` and show no such pair makes
`A · θ² + B · θ_★² = θ_retrain²` exact across all data choices in any
nonempty finite dimension `d`.

The proof reduces to the scalar Theorem 3b via *scalar-embedded data*. For
any chosen index `i₀ : d`, data of the form `α_c • 1` (scalar times the
identity) and `Pi.single i₀ β_c` (β_c on the i₀-th coordinate, 0 elsewhere)
makes every round-2 quantity equal to a scalar times `Pi.single i₀ 1`.
Projecting the matrix constraint onto the `i₀`-th coordinate yields the
scalar 3b constraint on `(A i₀ i₀, B i₀ i₀)` — which the scalar 3b says
has no solution.

The key algebraic lemma is `fp2_md_scalar_embed`: at scalar-embedded data,
every multi-dim round-2 quantity equals the corresponding scalar fp2 value
times the basis vector at `i₀`.
-/

import Theory.Impossibility
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

namespace FLUL

section MultiDimB

variable {d : Type*} [Fintype d] [DecidableEq d]

/-! ### Multi-dim FedProx round-2 quantities

Named extractions of the let-bindings inside `fedProx2_gradUnlearn_residual_md`
(`Theory.ImpossibilityMD`). The matrix `M_c := α_c + μ • 1` is the
regularized Gram matrix per client; `M_c⁻¹` is the junk-valued inverse,
agreeing with the true inverse when `α_c + μ • 1` is invertible. -/

/-- Multi-dim FedProx round-2 global aggregate. -/
noncomputable def fp2_θ2_md
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ) : d → ℝ :=
  let M_t := α_t + μ • (1 : Matrix d d ℝ)
  let M_r := α_r + μ • (1 : Matrix d d ℝ)
  let θ_t_1 := M_t⁻¹.mulVec β_t
  let θ_r_1 := M_r⁻¹.mulVec β_r
  let θ_1 := (1/2 : ℝ) • (θ_t_1 + θ_r_1)
  let θ_t_2 := M_t⁻¹.mulVec (β_t + μ • θ_1)
  let θ_r_2 := M_r⁻¹.mulVec (β_r + μ • θ_1)
  (1/2 : ℝ) • (θ_t_2 + θ_r_2)

/-- Multi-dim FedProx round-2 target's local update. -/
noncomputable def fp2_θt2_md
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ) : d → ℝ :=
  let M_t := α_t + μ • (1 : Matrix d d ℝ)
  let M_r := α_r + μ • (1 : Matrix d d ℝ)
  let θ_t_1 := M_t⁻¹.mulVec β_t
  let θ_r_1 := M_r⁻¹.mulVec β_r
  let θ_1 := (1/2 : ℝ) • (θ_t_1 + θ_r_1)
  M_t⁻¹.mulVec (β_t + μ • θ_1)

/-- Multi-dim FedProx round-2 retrain aggregate. -/
noncomputable def fp2_θretrain2_md
    (α_r : Matrix d d ℝ) (β_r : d → ℝ) (μ : ℝ) : d → ℝ :=
  let M_r := α_r + μ • (1 : Matrix d d ℝ)
  let θ_r_1 := M_r⁻¹.mulVec β_r
  M_r⁻¹.mulVec (β_r + μ • θ_r_1)

/-- A matrix pair `(A, B) ∈ Matrix d d ℝ × Matrix d d ℝ` is *exact* for
multi-dim FedProx-2-2 at `μ > 0` if `A · θ² + B · θ_★² = θ_retrain²` (as
vectors in `d → ℝ`) for every data choice. -/
def IsExactDataIndepLinear_FedProx2_md (A B : Matrix d d ℝ) (μ : ℝ) : Prop :=
  ∀ (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ),
    A.mulVec (fp2_θ2_md α_t α_r β_t β_r μ)
      + B.mulVec (fp2_θt2_md α_t α_r β_t β_r μ)
    = fp2_θretrain2_md α_r β_r μ

/-! ### Scalar-embedding reduction

For `c ≠ 0`, the regularized Gram matrix `c • 1 : Matrix d d ℝ` is invertible
with inverse `c⁻¹ • 1`. Combined with `Matrix.smul_mulVec_assoc` and
`Matrix.one_mulVec`, multiplication of `(c • 1)⁻¹` by a scaled basis vector
evaluates to a scaled basis vector. By induction, every round-2 quantity in
`fp2_*_md` at scalar-embedded data is a scaled basis vector. -/

/-- `(c • 1)⁻¹ = c⁻¹ • 1` for nonzero scalars `c`. -/
private lemma smul_one_inv (c : ℝ) (hc : c ≠ 0) :
    (c • (1 : Matrix d d ℝ))⁻¹ = c⁻¹ • (1 : Matrix d d ℝ) := by
  apply Matrix.inv_eq_left_inv
  rw [Matrix.smul_mul, Matrix.one_mul, smul_smul, inv_mul_cancel₀ hc, one_smul]

/-- `(c • 1).mulVec v = c • v`. -/
private lemma smul_one_mulVec (c : ℝ) (v : d → ℝ) :
    (c • (1 : Matrix d d ℝ)).mulVec v = c • v := by
  ext i
  simp [Matrix.mulVec, dotProduct, Matrix.smul_apply, Matrix.one_apply,
        Pi.smul_apply, smul_eq_mul, mul_ite, mul_one, mul_zero]

omit [Fintype d] in
/-- `(α • 1) + (μ • 1) = (α + μ) • 1`. -/
private lemma add_smul_one (α μ : ℝ) :
    (α • (1 : Matrix d d ℝ)) + μ • (1 : Matrix d d ℝ) = (α + μ) • (1 : Matrix d d ℝ) := by
  rw [← add_smul]

/-- The key reduction: at scalar-embedded data, each multi-dim round-2
quantity is the corresponding scalar fp2 value times the basis vector at i₀.

Bundled as one lemma so the three reductions (`fp2_θ2_md`, `fp2_θt2_md`,
`fp2_θretrain2_md`) share the same intermediate computations. -/
private lemma fp2_md_scalar_embed
    (i₀ : d) (α_t α_r β_t β_r μ : ℝ)
    (h_t : α_t + μ ≠ 0) (h_r : α_r + μ ≠ 0) :
    let e : d → ℝ := Pi.single i₀ 1
    fp2_θ2_md (α_t • (1 : Matrix d d ℝ)) (α_r • (1 : Matrix d d ℝ))
              (Pi.single i₀ β_t) (Pi.single i₀ β_r) μ
      = fp2_θ2 α_t α_r β_t β_r μ • e
    ∧ fp2_θt2_md (α_t • (1 : Matrix d d ℝ)) (α_r • (1 : Matrix d d ℝ))
                 (Pi.single i₀ β_t) (Pi.single i₀ β_r) μ
      = fp2_θt2 α_t α_r β_t β_r μ • e
    ∧ fp2_θretrain2_md (α_r • (1 : Matrix d d ℝ)) (Pi.single i₀ β_r) μ
      = fp2_θretrain2 α_r β_r μ • e := by
  -- Local basis vector at i₀.
  intro e
  -- Cache the regularized Gram matrices and their inverses on the scalar-embedded data.
  set M_t : Matrix d d ℝ := α_t • 1 + μ • 1 with hMt_def
  set M_r : Matrix d d ℝ := α_r • 1 + μ • 1 with hMr_def
  have hMt : M_t = (α_t + μ) • (1 : Matrix d d ℝ) := add_smul_one α_t μ
  have hMr : M_r = (α_r + μ) • (1 : Matrix d d ℝ) := add_smul_one α_r μ
  have hMt_inv : M_t⁻¹ = (α_t + μ)⁻¹ • (1 : Matrix d d ℝ) := by
    rw [hMt, smul_one_inv (α_t + μ) h_t]
  have hMr_inv : M_r⁻¹ = (α_r + μ)⁻¹ • (1 : Matrix d d ℝ) := by
    rw [hMr, smul_one_inv (α_r + μ) h_r]
  -- Pi.single i₀ β = β • e.
  have hβt : (Pi.single i₀ β_t : d → ℝ) = β_t • e := by
    ext j; by_cases h : j = i₀
    · subst h; simp [e]
    · simp [e, Pi.single_eq_of_ne h]
  have hβr : (Pi.single i₀ β_r : d → ℝ) = β_r • e := by
    ext j; by_cases h : j = i₀
    · subst h; simp [e]
    · simp [e, Pi.single_eq_of_ne h]
  -- Round-1 local updates: θ_c¹ = (β_c / (α_c + μ)) • e.
  have hθt1 : M_t⁻¹.mulVec (Pi.single i₀ β_t) = (β_t / (α_t + μ)) • e := by
    rw [hβt, hMt_inv, smul_one_mulVec, smul_smul, ← div_eq_inv_mul]
  have hθr1 : M_r⁻¹.mulVec (Pi.single i₀ β_r) = (β_r / (α_r + μ)) • e := by
    rw [hβr, hMr_inv, smul_one_mulVec, smul_smul, ← div_eq_inv_mul]
  -- Round-1 aggregate θ¹ = scalar • e, with scalar = (1/2) * (β_t/(α_t+μ) + β_r/(α_r+μ)).
  set θ_t_1_md : d → ℝ := M_t⁻¹.mulVec (Pi.single i₀ β_t) with hθt1_def
  set θ_r_1_md : d → ℝ := M_r⁻¹.mulVec (Pi.single i₀ β_r) with hθr1_def
  set θ_1_md : d → ℝ := (1/2 : ℝ) • (θ_t_1_md + θ_r_1_md) with hθ1_def
  have hθ1 : θ_1_md =
      ((1/2 : ℝ) * (β_t / (α_t + μ) + β_r / (α_r + μ))) • e := by
    rw [hθ1_def, hθt1, hθr1, ← add_smul, smul_smul]
  -- Round-2 target local update.
  have hθt2 : M_t⁻¹.mulVec (Pi.single i₀ β_t + μ • θ_1_md) =
      ((β_t + μ * ((1/2) * (β_t / (α_t + μ) + β_r / (α_r + μ)))) / (α_t + μ)) • e := by
    rw [hβt, hθ1, ← smul_assoc, ← add_smul, hMt_inv, smul_one_mulVec, smul_smul,
        ← div_eq_inv_mul]
    -- Inner `μ • X` (scalar smul on ℝ) vs `μ * X` is def-equal; congr 1 closes both subgoals.
    congr 1
  -- Round-2 retain local update.
  have hθr2 : M_r⁻¹.mulVec (Pi.single i₀ β_r + μ • θ_1_md) =
      ((β_r + μ * ((1/2) * (β_t / (α_t + μ) + β_r / (α_r + μ)))) / (α_r + μ)) • e := by
    rw [hβr, hθ1, ← smul_assoc, ← add_smul, hMr_inv, smul_one_mulVec, smul_smul,
        ← div_eq_inv_mul]
    congr 1
  -- Retrain trajectory round-2 (uses θ_r_1 in place of θ_1).
  have hθretrain2 : M_r⁻¹.mulVec (Pi.single i₀ β_r + μ • θ_r_1_md) =
      ((β_r + μ * (β_r / (α_r + μ))) / (α_r + μ)) • e := by
    rw [hβr, hθr1, ← smul_assoc, ← add_smul, hMr_inv, smul_one_mulVec, smul_smul,
        ← div_eq_inv_mul]
    congr 1
  refine ⟨?_, ?_, ?_⟩
  · -- fp2_θ2_md = fp2_θ2 • e
    change (1/2 : ℝ) • (M_t⁻¹.mulVec (Pi.single i₀ β_t + μ • θ_1_md)
                        + M_r⁻¹.mulVec (Pi.single i₀ β_r + μ • θ_1_md))
        = fp2_θ2 α_t α_r β_t β_r μ • e
    rw [hθt2, hθr2, ← add_smul, smul_smul]
    congr 1
    unfold fp2_θ2
    ring
  · -- fp2_θt2_md = fp2_θt2 • e
    change M_t⁻¹.mulVec (Pi.single i₀ β_t + μ • θ_1_md)
        = fp2_θt2 α_t α_r β_t β_r μ • e
    rw [hθt2]
    congr 1
    unfold fp2_θt2
    ring
  · -- fp2_θretrain2_md = fp2_θretrain2 • e (congr 1 closes via def-equality)
    change M_r⁻¹.mulVec (Pi.single i₀ β_r + μ • θ_r_1_md)
        = fp2_θretrain2 α_r β_r μ • e
    rw [hθretrain2]
    congr 1

/-- `A.mulVec (s • Pi.single i₀ 1)` at coordinate `i₀` equals `s * A i₀ i₀`. -/
private lemma mulVec_smul_single_apply
    (A : Matrix d d ℝ) (i₀ : d) (s : ℝ) :
    (A.mulVec (s • (Pi.single i₀ 1 : d → ℝ))) i₀ = s * A i₀ i₀ := by
  rw [Matrix.mulVec_smul, Pi.smul_apply, smul_eq_mul]
  congr 1
  change (A i₀) ⬝ᵥ (Pi.single i₀ (1 : ℝ)) = A i₀ i₀
  rw [dotProduct_single, mul_one]

/-- **Multi-dim Theorem 3b**: for every `μ > 0` and every `i₀ : d`, no matrix
pair `(A, B) : Matrix d d ℝ × Matrix d d ℝ` is exact for multi-dim
FedProx-2-2 across all data choices.

Proof. Suppose `(A, B)` is exact. The matrix witness gives a scalar witness
`(A i₀ i₀, B i₀ i₀)` for scalar 3b, contradicting
`fedProx2_no_exact_data_indep_linear`.

For any scalar data `(α_t, α_r, β_t, β_r)` with `0 ≤ α_t, α_r`, apply the
matrix-3b predicate at scalar-embedded data
`(α_t • 1, α_r • 1, Pi.single i₀ β_t, Pi.single i₀ β_r)`. The reduction
`fp2_md_scalar_embed` rewrites each multi-dim round-2 quantity as the
scalar fp2 value times `Pi.single i₀ 1`. Evaluating the matrix-3b equation
at coordinate `i₀` yields exactly scalar 3b's constraint on `(A i₀ i₀, B i₀ i₀)`. -/
theorem fedProx2_no_exact_data_indep_linear_md
    (i₀ : d) (μ : ℝ) (hμ : 0 < μ) :
    ¬ ∃ A B : Matrix d d ℝ, IsExactDataIndepLinear_FedProx2_md A B μ := by
  rintro ⟨A, B, h⟩
  apply fedProx2_no_exact_data_indep_linear μ hμ
  refine ⟨A i₀ i₀, B i₀ i₀, ?_⟩
  intro α_t α_r β_t β_r hα_t hα_r
  have h_t_ne : α_t + μ ≠ 0 := by positivity
  have h_r_ne : α_r + μ ≠ 0 := by positivity
  have hMd := h (α_t • 1) (α_r • 1) (Pi.single i₀ β_t) (Pi.single i₀ β_r)
  obtain ⟨hθ2, hθt2, hθretrain2⟩ :=
    fp2_md_scalar_embed i₀ α_t α_r β_t β_r μ h_t_ne h_r_ne
  rw [hθ2, hθt2, hθretrain2] at hMd
  -- Evaluate the vector equation at coordinate i₀.
  have hMd_i₀ := congrFun hMd i₀
  rw [Pi.add_apply, mulVec_smul_single_apply, mulVec_smul_single_apply] at hMd_i₀
  -- hMd_i₀ : fp2_θ2 * A i₀ i₀ + fp2_θt2 * B i₀ i₀ = (fp2_θretrain2 • Pi.single i₀ 1) i₀
  -- RHS: (scalar • Pi.single i₀ 1) i₀ = scalar.
  rw [Pi.smul_apply, smul_eq_mul, Pi.single_eq_same, mul_one] at hMd_i₀
  -- Rearrange to match scalar 3b's predicate.
  linarith [hMd_i₀]

end MultiDimB

end FLUL
