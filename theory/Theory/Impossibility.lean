/-
# Impossibility Theorems 3a and 3b (L.4)

Two impossibility results for 2-client, T = 2 FedProx with equal weights.

* **Theorem 3a** (`fedProx2_gradUnlearn_residual`, L.4a): closed-form
  gradient-subtraction residual. Matches `run_e3_impossibility_toy.py` to
  within `7.89 × 10⁻¹⁷` over 1 000 random trials.
* **Theorem 3b** (`fedProx2_no_exact_data_indep_linear`, L.4b): for every
  `μ > 0`, no *data-independent linear* operator
  `Λ(θ², θ_★², 1, 2) = a · θ² + b · θ_★²` with fixed `(a, b) ∈ ℝ²` is exact
  across all data choices. Witnessed by three parametric
  `(α_★, α_r, β_★, β_r)` tuples that scale with `μ` so the induced
  constraints collapse to a μ-invariant integer linear system.

E.3 corroborates 3b empirically over `μ ∈ [0.001, 100]`.
-/

import Theory.FL
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity

namespace FLUL

/-- **Theorem 3a**: closed-form gradient-subtraction residual for 2-client,
T = 2 FedProx with equal weights (`w_target = w_retain = 1`, `W = 2`).

Setup: each client `c ∈ {target, retain}` has a closed-form FedProx local
solver
    `θ_c^(t) = (β_c + μ · θ^(t-1)) / (α_c + μ)`
with `θ^(0) = 0`. FedAvg aggregator on the two clients. The retrain
trajectory has only the retain client (so `θ_retrain^(t) = θ_retain` under
retrain dynamics).

Statement: at round 2,
    `gradUnlearn(θ², θ_target², 1, 2) − θ_retrain² = δ_grad`
where
    `δ_grad = (μ / (2 · (α_retain + μ))) · (θ_target¹ − θ_retain¹)`.

The residual factors into a `μ`-coupling term (monotone in `μ`) and a
round-1 client disagreement term. The latter shrinks with `μ`, producing
the non-monotone shape observed in E.2 and E.3.

Empirical corroboration: E.3 verifies this identity to `≈ 10⁻¹⁶` across
1 000 trials per `μ`. -/
theorem fedProx2_gradUnlearn_residual
    (α_target α_retain β_target β_retain μ : ℝ)
    (hα_target_nn : 0 ≤ α_target)
    (hα_retain_nn : 0 ≤ α_retain)
    (hμ_pos : 0 < μ) :
    let θ_t_1 := β_target / (α_target + μ)
    let θ_r_1 := β_retain / (α_retain + μ)
    let θ_1 := (θ_t_1 + θ_r_1) / 2
    let θ_t_2 := (β_target + μ * θ_1) / (α_target + μ)
    let θ_r_2 := (β_retain + μ * θ_1) / (α_retain + μ)
    let θ_2 := (θ_t_2 + θ_r_2) / 2
    let θ_retrain_2 := (β_retain + μ * θ_r_1) / (α_retain + μ)
    gradUnlearn θ_2 θ_t_2 1 2 - θ_retrain_2 =
      μ / (2 * (α_retain + μ)) * (θ_t_1 - θ_r_1) := by
  intro θ_t_1 θ_r_1 θ_1 θ_t_2 θ_r_2 θ_2 θ_retrain_2
  unfold gradUnlearn
  have hαt : α_target + μ > 0 := by linarith
  have hαr : α_retain + μ > 0 := by linarith
  have hαt_ne : α_target + μ ≠ 0 := ne_of_gt hαt
  have hαr_ne : α_retain + μ ≠ 0 := ne_of_gt hαr
  simp only [θ_2, θ_t_2, θ_r_2, θ_1, θ_t_1, θ_r_1, θ_retrain_2]
  field_simp
  ring

/-- **Theorem 3a, scalar form**: the residual is strictly positive whenever
`μ > 0` and the two clients' round-1 closed-form solutions disagree.

This is the "no-vanishing" statement from `theorems_draft.tex` Theorem 3a's
conclusion. -/
theorem fedProx2_gradUnlearn_residual_nonzero
    (α_target α_retain β_target β_retain μ : ℝ)
    (hα_target_nn : 0 ≤ α_target)
    (hα_retain_nn : 0 ≤ α_retain)
    (hμ_pos : 0 < μ)
    (hdisagree :
      β_target / (α_target + μ) ≠ β_retain / (α_retain + μ)) :
    let θ_t_1 := β_target / (α_target + μ)
    let θ_r_1 := β_retain / (α_retain + μ)
    let θ_1 := (θ_t_1 + θ_r_1) / 2
    let θ_t_2 := (β_target + μ * θ_1) / (α_target + μ)
    let θ_r_2 := (β_retain + μ * θ_1) / (α_retain + μ)
    let θ_2 := (θ_t_2 + θ_r_2) / 2
    let θ_retrain_2 := (β_retain + μ * θ_r_1) / (α_retain + μ)
    gradUnlearn θ_2 θ_t_2 1 2 - θ_retrain_2 ≠ 0 := by
  intro _ _ _ _ _ _ _
  rw [fedProx2_gradUnlearn_residual α_target α_retain β_target β_retain μ
        hα_target_nn hα_retain_nn hμ_pos]
  apply mul_ne_zero
  · -- μ / (2 * (α_retain + μ)) ≠ 0
    apply ne_of_gt
    apply div_pos hμ_pos
    nlinarith
  · -- (θ_t_1 - θ_r_1) ≠ 0  — let-bindings unfold to the explicit form
    change β_target / (α_target + μ) - β_retain / (α_retain + μ) ≠ 0
    intro h
    exact hdisagree (sub_eq_zero.mp h)

/-! ## Strong Impossibility — Theorem 3b (L.4b)

`theorems_draft.tex` Theorem 3b states that for every `μ > 0`, no
*data-independent linear* operator `Λ(θ², θ_★², 1, 2) = a · θ² + b · θ_★²`
with fixed `(a, b) ∈ ℝ²` is exact for FedProx-2-2 across all data choices.

We prove the general `μ > 0` statement via a *parametric* three-instance
witness whose constraint equations are μ-invariant: the round-2 quantities
all scale as `1/μ`, so after clearing denominators the three constraints
collapse to the same integer system as the `μ = 1` case in
`theorems_draft.tex`. Instances:
* `(α_★, α_r, β_★, β_r) = (μ, μ, 1, 0)` ↦ constraint `3 a + 5 b = 0`.
* `(α_★, α_r, β_★, β_r) = (μ, μ, 0, 1)` ↦ constraint `3 a + b = 6`.
* `(α_★, α_r, β_★, β_r) = (2μ, μ, 1, 1)` ↦ constraint `85 a + 68 b = 108`.
Instances 1 and 2 pin `(a, b) = (5/2, -3/2)`, which fails instance 3 by `2.5`.

Empirical corroboration: E.3 (`run_e3_impossibility_toy.py`) finds the
OLS-optimal data-independent `(a, b)` leaves nonzero residual at every
`μ > 0` sampled. -/

/-- Round-2 FedProx-2 global aggregate (`θ²`), equal weights, `θ⁰ = 0`. -/
noncomputable def fp2_θ2
    (α_target α_retain β_target β_retain μ : ℝ) : ℝ :=
  let θ_t_1 := β_target / (α_target + μ)
  let θ_r_1 := β_retain / (α_retain + μ)
  let θ_1 := (θ_t_1 + θ_r_1) / 2
  let θ_t_2 := (β_target + μ * θ_1) / (α_target + μ)
  let θ_r_2 := (β_retain + μ * θ_1) / (α_retain + μ)
  (θ_t_2 + θ_r_2) / 2

/-- Round-2 FedProx-2 target client's local update (`θ_★²`). -/
noncomputable def fp2_θt2
    (α_target α_retain β_target β_retain μ : ℝ) : ℝ :=
  let θ_t_1 := β_target / (α_target + μ)
  let θ_r_1 := β_retain / (α_retain + μ)
  let θ_1 := (θ_t_1 + θ_r_1) / 2
  (β_target + μ * θ_1) / (α_target + μ)

/-- Round-2 FedProx retrain trajectory aggregate (`θ_retrain²`).
Only the retain client participates (target's contribution removed). -/
noncomputable def fp2_θretrain2
    (α_retain β_retain μ : ℝ) : ℝ :=
  let θ_r_1 := β_retain / (α_retain + μ)
  (β_retain + μ * θ_r_1) / (α_retain + μ)

/-- A pair `(a, b)` is *exact* for FedProx-2-2 at proximal coefficient `μ`
if the data-independent linear combination `a · θ² + b · θ_★²` equals the
retrain aggregate `θ_retrain²` for every data choice with `α_★, α_r ≥ 0`.

Cf. `theorems_draft.tex` Theorem 3b statement. -/
def IsExactDataIndepLinear_FedProx2 (a b μ : ℝ) : Prop :=
  ∀ α_target α_retain β_target β_retain : ℝ,
    0 ≤ α_target → 0 ≤ α_retain →
    a * fp2_θ2 α_target α_retain β_target β_retain μ
      + b * fp2_θt2 α_target α_retain β_target β_retain μ
    = fp2_θretrain2 α_retain β_retain μ

/-! ### Parametric three-instance witness

The three instances scale with `μ` so the round-2 quantities all carry an
explicit factor of `1/μ`. After clearing denominators, the constraint
equations become identical to the `μ = 1` integer system. -/

/-- Parametric instance 1, `θ²`: `(α_★, α_r, β_★, β_r) = (μ, μ, 1, 0)` ↦ `θ² = 3/(8μ)`. -/
lemma fp2_paramInstance1_θ2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θ2 μ μ 1 0 μ = 3 / (8 * μ) := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θ2; field_simp; ring

/-- Parametric instance 1, `θ_★²`: yields `5/(8μ)`. -/
lemma fp2_paramInstance1_θt2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θt2 μ μ 1 0 μ = 5 / (8 * μ) := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θt2; field_simp; ring

/-- Parametric instance 1, `θ_retrain²`: yields `0`. -/
lemma fp2_paramInstance1_θretrain2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θretrain2 μ 0 μ = 0 := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θretrain2; field_simp; ring

/-- Parametric instance 2, `θ²`: `(α_★, α_r, β_★, β_r) = (μ, μ, 0, 1)` ↦ `θ² = 3/(8μ)`. -/
lemma fp2_paramInstance2_θ2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θ2 μ μ 0 1 μ = 3 / (8 * μ) := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θ2; field_simp; ring

/-- Parametric instance 2, `θ_★²`: yields `1/(8μ)`. -/
lemma fp2_paramInstance2_θt2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θt2 μ μ 0 1 μ = 1 / (8 * μ) := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θt2; field_simp; ring

/-- Parametric instance 2, `θ_retrain²`: yields `3/(4μ)`.
Reused by instance 3 (same retain data). -/
lemma fp2_paramInstance2_θretrain2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θretrain2 μ 1 μ = 3 / (4 * μ) := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θretrain2; field_simp; ring

/-- Parametric instance 3, `θ²`: `(α_★, α_r, β_★, β_r) = (2μ, μ, 1, 1)` ↦ `θ² = 85/(144μ)`. -/
lemma fp2_paramInstance3_θ2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θ2 (2 * μ) μ 1 1 μ = 85 / (144 * μ) := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θ2; field_simp; ring

/-- Parametric instance 3, `θ_★²`: yields `17/(36μ)`. -/
lemma fp2_paramInstance3_θt2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θt2 (2 * μ) μ 1 1 μ = 17 / (36 * μ) := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θt2; field_simp; ring

/-- **Theorem 3b**: for every `μ > 0`, no data-independent linear unlearning
operator `Λ(θ², θ_★², 1, 2) = a · θ² + b · θ_★²` is exact for FedProx-2-2
across all data choices.

Proof. Suppose `(a, b)` is exact at some fixed `μ > 0`. Specializing the
predicate to the three parametric instances above and rewriting with the
parametric value lemmas gives
* `a · (3/(8μ)) + b · (5/(8μ)) = 0`         (instance 1, retrain = 0)
* `a · (3/(8μ)) + b · (1/(8μ)) = 3/(4μ)`    (instance 2)
* `a · (85/(144μ)) + b · (17/(36μ)) = 3/(4μ)`  (instance 3)
`field_simp` clears denominators (using `μ ≠ 0`), collapsing the system to
the μ-invariant integer constraints
* `3 a + 5 b = 0`
* `3 a + b = 6`
* `85 a + 68 b = 108`
The first two pin `(a, b) = (5/2, -3/2)`; substituting into the third gives
`85·(5/2) + 68·(-3/2) = 110.5 ≠ 108`. `linarith` finds this contradiction. -/
theorem fedProx2_no_exact_data_indep_linear (μ : ℝ) (hμ : 0 < μ) :
    ¬ ∃ a b : ℝ, IsExactDataIndepLinear_FedProx2 a b μ := by
  rintro ⟨a, b, h⟩
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  -- Instance 1: (μ, μ, 1, 0)
  have h1 := h μ μ 1 0 hμ.le hμ.le
  rw [fp2_paramInstance1_θ2 μ hμ, fp2_paramInstance1_θt2 μ hμ,
      fp2_paramInstance1_θretrain2 μ hμ] at h1
  -- Instance 2: (μ, μ, 0, 1)
  have h2 := h μ μ 0 1 hμ.le hμ.le
  rw [fp2_paramInstance2_θ2 μ hμ, fp2_paramInstance2_θt2 μ hμ,
      fp2_paramInstance2_θretrain2 μ hμ] at h2
  -- Instance 3: (2μ, μ, 1, 1) — retrain reuses instance 2's (same retain data)
  have h3 := h (2 * μ) μ 1 1 (by positivity) hμ.le
  rw [fp2_paramInstance3_θ2 μ hμ, fp2_paramInstance3_θt2 μ hμ,
      fp2_paramInstance2_θretrain2 μ hμ] at h3
  -- Clear denominators: integer system `3a+5b=0, 3a+b=6, 85a+68b=108`.
  field_simp at h1 h2 h3
  linarith

end FLUL
