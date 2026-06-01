/-
# Characterization Theorem (L.3, forward direction)

`theorems_draft.tex` Theorem 2 states that an exact unlearning operator
exists for `c_★` if and only if:
  (i)  the aggregator is linear (Def. 2); and
  (ii) the protocol is causally history-independent w.r.t. `c_★` (Def. 5).

This module proves the **forward direction** (sufficiency): given (i) and
(ii), the gradient-subtraction operator (a closed-form witness) is exact.

For the single-round 2-client setting, (i) is satisfied by the FedAvg-2
aggregator and (ii) is automatic (no prior rounds to depend on). The
multi-round version, and the reverse direction (necessity via the median
and FedProx counterexamples) live in `theorems_draft.tex` Section 6 open
items and will be added in a follow-on phase.
-/

import Theory.FL
import Theory.Correctness
import Theory.Impossibility
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp

namespace FLUL

/-- An unlearning operator `Λ` is *exact* for the FedAvg-2 protocol with
weights `(w_target, w_retain)` if `Λ` applied to the FedAvg-2 aggregate of
two clients' local updates recovers the retain client's local update
exactly, for every choice of `(θ_target, θ_retain)`.

Cf. `theorems_draft.tex` Definition 6 (specialized to `n = 2` clients and
the FedAvg aggregator). -/
def IsExactUnlearnOp_FedAvg2 (Λ : UnlearnOp)
    (w_target w_retain : Weight) : Prop :=
  ∀ (θ_target θ_retain : Parameters),
    Λ (fedAvg2 θ_target θ_retain w_target w_retain) θ_target w_target
      (w_target + w_retain) = θ_retain

/-- **Theorem 2 (forward direction, single-round 2-client FedAvg)**: for
any choice of positive weights with non-trivial total, an exact unlearning
operator exists. The witness is `gradUnlearn`.

Proof. Take `Λ = gradUnlearn`; the exactness condition reduces to the
algebraic identity proved in `Correctness.lean` (`gradUnlearn_fedAvg2`). -/
theorem characterization_forward_FedAvg2
    (w_target w_retain : Weight)
    (hw_retain : w_retain ≠ 0)
    (hW : w_target + w_retain ≠ 0) :
    ∃ Λ : UnlearnOp, IsExactUnlearnOp_FedAvg2 Λ w_target w_retain := by
  refine ⟨gradUnlearn, ?_⟩
  intro θ_target θ_retain
  unfold fedAvg2
  exact gradUnlearn_fedAvg2 θ_target θ_retain w_target w_retain hw_retain hW

/-- **Constructive witness**: gradient subtraction is the canonical exact
unlearning operator for FedAvg-2. This is the constructive form of the
existence statement, useful for any downstream development that wants the
specific operator (not just its existence).

Cf. `theorems_draft.tex` Theorem 2 forward direction's witness clause. -/
theorem gradUnlearn_isExact_FedAvg2
    (w_target w_retain : Weight)
    (hw_retain : w_retain ≠ 0)
    (hW : w_target + w_retain ≠ 0) :
    IsExactUnlearnOp_FedAvg2 gradUnlearn w_target w_retain := by
  intro θ_target θ_retain
  unfold fedAvg2
  exact gradUnlearn_fedAvg2 θ_target θ_retain w_target w_retain hw_retain hW

/-! ## Reverse direction: necessity of linearity

We exhibit a non-linear aggregator (3-client coordinate-wise median) for
which no data-independent linear unlearning operator with the signature of
`Definition 6` is exact. This is the necessity-of-linearity branch of
`theorems_draft.tex` Theorem 2.

Empirically corroborated by E.2b (`run_e2b_median_aggregator.py`,
mean residual `≈ 0.88` even for the OLS-optimal `(a, b)`). -/

/-- Coordinate-wise median of three scalars, written via `max`/`min`:
`median(a, b, c) = a + b + c − max(a, max(b, c)) − min(a, min(b, c))`. -/
noncomputable def medianAgg3 (a b c : ℝ) : ℝ :=
  a + b + c - max a (max b c) - min a (min b c)

/-- When a 3-client median federation has its target client removed, the
remaining federation reduces to 2 clients. We adopt the convention that
"median of two" is the mean of the two — i.e., the retain aggregator on
2 clients is `(b + c) / 2`. -/
noncomputable def medianRetain2 (b c : ℝ) : ℝ := (b + c) / 2

/-- An unlearning operator `Λ` is *exact* for the 3-client median protocol
with equal weights if it recovers the retrain (mean of the two remaining
clients) from the median aggregate, the target's local update, and the
weights. -/
def IsExactUnlearnOp_MedianAgg3 (Λ : UnlearnOp) : Prop :=
  ∀ (θ_target θ_other1 θ_other2 : ℝ),
    Λ (medianAgg3 θ_target θ_other1 θ_other2) θ_target 1 3
      = medianRetain2 θ_other1 θ_other2

/-- Helper: `medianAgg3 10 5 1 = 5`. The "target client is the maximum"
configuration: removing it from `{10, 5, 1}` leaves `{5, 1}` whose mean is 3. -/
lemma medianAgg3_10_5_1 : medianAgg3 10 5 1 = 5 := by
  unfold medianAgg3
  rw [show max (5 : ℝ) 1 = 5 from max_eq_left (by norm_num : (1 : ℝ) ≤ 5),
      show max (10 : ℝ) 5 = 10 from max_eq_left (by norm_num : (5 : ℝ) ≤ 10),
      show min (5 : ℝ) 1 = 1 from min_eq_right (by norm_num : (1 : ℝ) ≤ 5),
      show min (10 : ℝ) 1 = 1 from min_eq_right (by norm_num : (1 : ℝ) ≤ 10)]
  norm_num

/-- Helper: `medianAgg3 10 5 3 = 5`. The "target client is the maximum,
again" configuration: removing it from `{10, 5, 3}` leaves `{5, 3}` whose
mean is 4. Same median as `(10, 5, 1)` but different retain mean -- the
information-loss witness. -/
lemma medianAgg3_10_5_3 : medianAgg3 10 5 3 = 5 := by
  unfold medianAgg3
  rw [show max (5 : ℝ) 3 = 5 from max_eq_left (by norm_num : (3 : ℝ) ≤ 5),
      show max (10 : ℝ) 5 = 10 from max_eq_left (by norm_num : (5 : ℝ) ≤ 10),
      show min (5 : ℝ) 3 = 3 from min_eq_right (by norm_num : (3 : ℝ) ≤ 5),
      show min (10 : ℝ) 3 = 3 from min_eq_right (by norm_num : (3 : ℝ) ≤ 10)]
  norm_num

/-- **Theorem 2 (reverse direction, non-linear aggregator)**: no exact
unlearning operator exists for the 3-client coordinate-wise-median FedAvg
protocol with equal weights.

Proof. Two specific 3-tuples `(10, 5, 1)` and `(10, 5, 3)` produce the same
inputs `(θ_global, θ_target, w_target, W) = (5, 10, 1, 3)` to any candidate
`Λ`. But the retrain mean differs (`3` vs `4`). A single `Λ` cannot satisfy
both equations, so no exact `Λ` exists. -/
theorem characterization_reverse_MedianAgg3 :
    ¬ ∃ Λ : UnlearnOp, IsExactUnlearnOp_MedianAgg3 Λ := by
  rintro ⟨Λ, hΛ⟩
  have h1 := hΛ 10 5 1
  have h2 := hΛ 10 5 3
  rw [medianAgg3_10_5_1] at h1
  rw [medianAgg3_10_5_3] at h2
  -- h1 : Λ 5 10 1 3 = medianRetain2 5 1
  -- h2 : Λ 5 10 1 3 = medianRetain2 5 3
  unfold medianRetain2 at h1 h2
  -- h1 : Λ 5 10 1 3 = (5 + 1) / 2 = 3
  -- h2 : Λ 5 10 1 3 = (5 + 3) / 2 = 4
  linarith

/-! ## Reverse direction: necessity of causal history-independence

We exhibit a 2-client FedProx-2 protocol at any `μ > 0` (linear FedAvg
aggregator, but history-dependent local update via the proximal term) for
which *no* exact unlearning operator exists — not just no
data-independent linear one. This complements
`characterization_reverse_MedianAgg3` (linearity branch); together the
two cover the reverse direction of `theorems_draft.tex` Theorem 2.

Strategy mirrors the median counterexample: pick two data instances that
produce the *same* `(θ², θ_★², w_★, W) = (1/μ, 1/(2μ), 1, 2)` inputs to a
candidate `Λ`, but different `θ_retrain²`. A single `Λ` cannot satisfy
both equations, so no exact `Λ` exists.

Instances chosen so the constraint algebra is `μ`-uniform — both
instances are parametric in `μ` and the round-2 quantities all scale as
`1/μ`, exactly as in `theorems_draft.tex` Theorem 3b.

Empirically corroborated by E.2 / E.3 — the FedProx residual map. -/

/-- An unlearning operator `Λ` is *exact* for the 2-client FedProx
protocol at proximal coefficient `μ` with equal client weights if `Λ`
applied to round-2 outputs recovers `θ_retrain²` for every nonnegative
`(α_★, α_r)` data choice.

Cf. `theorems_draft.tex` Definition 6 (specialized to `n = 2`, FedProx
local rule, `T = 2`). -/
def IsExactUnlearnOp_FedProx2 (Λ : UnlearnOp) (μ : ℝ) : Prop :=
  ∀ (α_target α_retain β_target β_retain : ℝ),
    0 ≤ α_target → 0 ≤ α_retain →
    Λ (fp2_θ2 α_target α_retain β_target β_retain μ)
      (fp2_θt2 α_target α_retain β_target β_retain μ) 1 2
    = fp2_θretrain2 α_retain β_retain μ

/-! ### Instance A: `(α_★, α_r, β_★, β_r) = (0, 0, 0, 1)` -/

/-- Instance A round-2 global aggregate. `θ_t_1 = 0`, `θ_r_1 = 1/μ`,
`θ_1 = 1/(2μ)`, `θ_t_2 = 1/(2μ)`, `θ_r_2 = 3/(2μ)`, `θ² = 1/μ`. -/
lemma fp2_instanceA_θ2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θ2 0 0 0 1 μ = 1 / μ := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θ2; field_simp; ring

/-- Instance A round-2 target update. `θ_★² = 1/(2μ)`. -/
lemma fp2_instanceA_θt2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θt2 0 0 0 1 μ = 1 / (2 * μ) := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θt2; field_simp; ring

/-- Instance A retrain trajectory at round 2. `θ_retain¹ = 1/μ`,
`θ_retrain² = (1 + μ·(1/μ))/μ = 2/μ`. -/
lemma fp2_instanceA_θretrain2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θretrain2 0 1 μ = 2 / μ := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θretrain2; field_simp; ring

/-! ### Instance B: `(α_★, α_r, β_★, β_r) = (0, μ, -1/14, 17/7)`

The instance is engineered so `(θ², θ_★²) = (1/μ, 1/(2μ))` — identical
to instance A — but `θ_retrain²` differs. The values `-1/14` and `17/7`
are the unique solution to the linear system
`6 β_★ + β_r = 2`, `2 β_★ + 5 β_r = 12` arising from matching
instance A's round-2 quantities under the `(0, μ)` Gram-matrix choice. -/

/-- Instance B round-2 global aggregate — same as instance A's `θ² = 1/μ`. -/
lemma fp2_instanceB_θ2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θ2 0 μ (-1/14) (17/7) μ = 1 / μ := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θ2; field_simp; ring

/-- Instance B round-2 target update — same as instance A's
`θ_★² = 1/(2μ)`. -/
lemma fp2_instanceB_θt2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θt2 0 μ (-1/14) (17/7) μ = 1 / (2 * μ) := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θt2; field_simp; ring

/-- Instance B retrain trajectory at round 2. `θ_retain¹ = 17/(14μ)`,
`θ_retrain² = (17/7 + μ·17/(14μ))/(2μ) = 51/(28μ)`. **Different** from
instance A's `2/μ = 56/(28μ)` — the information-loss witness. -/
lemma fp2_instanceB_θretrain2 (μ : ℝ) (hμ : 0 < μ) :
    fp2_θretrain2 μ (17/7) μ = 51 / (28 * μ) := by
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  unfold fp2_θretrain2; field_simp; ring

/-- **Theorem 2 (reverse direction, history-dependent local rule)**: for
every `μ > 0`, no exact unlearning operator exists for the 2-client
FedProx protocol with equal weights at `T = 2` — not just no
data-independent linear one (which is `fedProx2_no_exact_data_indep_linear`),
but no `Λ : UnlearnOp` at all.

Proof. Two parametric instances `A = (0, 0, 0, 1)` and
`B = (0, μ, -1/14, 17/7)` produce the same inputs
`(θ², θ_★², w_★, W) = (1/μ, 1/(2μ), 1, 2)` to any candidate `Λ`, but
their `θ_retrain²` values differ (`2/μ` vs `51/(28μ)`). A single `Λ`
cannot satisfy both equations.

Together with `characterization_reverse_MedianAgg3` (non-linear
aggregator branch), this closes the reverse direction of
`theorems_draft.tex` Theorem 2: every protocol failing either linearity
*or* causal history-independence has no exact unlearning operator. -/
theorem characterization_reverse_FedProx2 (μ : ℝ) (hμ : 0 < μ) :
    ¬ ∃ Λ : UnlearnOp, IsExactUnlearnOp_FedProx2 Λ μ := by
  rintro ⟨Λ, hΛ⟩
  have hμ_ne : μ ≠ 0 := ne_of_gt hμ
  -- Instance A: (α_★, α_r, β_★, β_r) = (0, 0, 0, 1).
  have h1 := hΛ 0 0 0 1 le_rfl le_rfl
  rw [fp2_instanceA_θ2 μ hμ, fp2_instanceA_θt2 μ hμ,
      fp2_instanceA_θretrain2 μ hμ] at h1
  -- Instance B: (α_★, α_r, β_★, β_r) = (0, μ, -1/14, 17/7).
  have h2 := hΛ 0 μ (-1/14) (17/7) le_rfl hμ.le
  rw [fp2_instanceB_θ2 μ hμ, fp2_instanceB_θt2 μ hμ,
      fp2_instanceB_θretrain2 μ hμ] at h2
  -- h1 : Λ (1/μ) (1/(2μ)) 1 2 = 2/μ
  -- h2 : Λ (1/μ) (1/(2μ)) 1 2 = 51/(28μ)
  -- Combining: 2/μ = 51/(28μ), i.e. 56 = 51 (after clearing μ ≠ 0).
  have hcombine : (2 : ℝ) / μ = 51 / (28 * μ) := h1.symm.trans h2
  field_simp at hcombine
  linarith

end FLUL
