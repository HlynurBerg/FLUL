/-
# Multi-Dim Multi-Round Theorem 1 (L.2-MR-MD)

Lift the scalar multi-round correctness theorem (`CorrectnessMR.lean`) to
arbitrary finite-dimensional parameter spaces. The implementation is
polymorphic over any `ℝ`-module `P`, so the scalar case (`P = ℝ`) and the
multi-dim case (`P = d → ℝ` for `[Fintype d] [DecidableEq d]`) are both
instances.

Setup. Two clients with a shared local update rule `U : P → D → P`,
datasets `D_target, D_retain : D` (arbitrary dataset type), scalar weights
`w_target, w_retain : ℝ`, and initial parameter `θ_init : P`. The original
2-client FedAvg trajectory and retrain trajectory are defined recursively
in `P`. The aggregator and unlearning operator use `ℝ`-smul on `P`:

* `fedAvg2_mod (θ_t θ_r : P) (w_t w_r : ℝ) := (1/(w_t + w_r)) • (w_t • θ_t + w_r • θ_r)`
* `gradUnlearn_mod (θ θ_t : P) (w_t W : ℝ) := (1/(W - w_t)) • (W • θ - w_t • θ_t)`

Result. Under FedAvg-2 and *causal* history-independence
(`theorems_draft.tex` Def. 5), gradient subtraction at round `n + 1`
recovers the retrain trajectory at round `n + 1`. Strong history-independence
implies causal-HI, so the strong-HI form follows.

The proof structurally mirrors the scalar version in `CorrectnessMR.lean`:
single-round identity (`gradUnlearn_fedAvg2_mod`) + one application of
causal-HI converts `U(trajF n) D_retain` to `trajR (n + 1)`.
-/

import Theory.FL
import Mathlib.Algebra.Module.Basic

namespace FLUL

namespace MultiRoundMD

variable {P : Type*} [AddCommGroup P] [Module ℝ P]
variable {D : Type*}

/-! ### Generic FedAvg-2 and gradient-subtraction operators -/

/-- FedAvg-2 over an `ℝ`-module: `(1/(w_t + w_r)) • (w_t • θ_t + w_r • θ_r)`. -/
noncomputable def fedAvg2_mod (θ_t θ_r : P) (w_t w_r : ℝ) : P :=
  (1 / (w_t + w_r)) • (w_t • θ_t + w_r • θ_r)

/-- Gradient-subtraction unlearning over an `ℝ`-module:
`(1/(W - w_target)) • (W • θ - w_target • θ_target)`. -/
noncomputable def gradUnlearn_mod (θ θ_target : P) (w_target W : ℝ) : P :=
  (1 / (W - w_target)) • (W • θ - w_target • θ_target)

/-! ### Single-round algebraic core -/

/-- **Theorem 1, algebraic core, module-polymorphic**: given the FedAvg-2
aggregate of two clients (`θ_t, θ_r ∈ P`) with positive total weight and
nonzero retain weight, gradient subtraction recovers `θ_r` exactly. -/
theorem gradUnlearn_fedAvg2_mod
    (θ_t θ_r : P) (w_t w_r : ℝ)
    (hw_r : w_r ≠ 0) (hW : w_t + w_r ≠ 0) :
    gradUnlearn_mod (fedAvg2_mod θ_t θ_r w_t w_r) θ_t w_t (w_t + w_r) = θ_r := by
  unfold gradUnlearn_mod fedAvg2_mod
  have h_diff : w_t + w_r - w_t = w_r := by ring
  have h1 : (w_t + w_r) * (1 / (w_t + w_r)) = 1 := by field_simp
  have h2 : (1 / w_r) * w_r = 1 := by field_simp
  rw [h_diff, smul_smul, h1, one_smul, add_sub_cancel_left, smul_smul, h2, one_smul]

/-! ### Trajectories -/

/-- One step of the 2-client FedAvg-mod protocol. -/
noncomputable def step (U : P → D → P)
    (D_target D_retain : D) (w_target w_retain : ℝ) (θ_prev : P) : P :=
  fedAvg2_mod (U θ_prev D_target) (U θ_prev D_retain) w_target w_retain

/-- Original 2-client FedAvg-mod trajectory. -/
noncomputable def trajF (U : P → D → P)
    (D_target D_retain : D) (w_target w_retain : ℝ) (θ_init : P) : ℕ → P
  | 0 => θ_init
  | n + 1 =>
      step U D_target D_retain w_target w_retain
        (trajF U D_target D_retain w_target w_retain θ_init n)

/-- Retrain trajectory: only the retain client participates. -/
noncomputable def trajR (U : P → D → P) (D_retain : D) (θ_init : P) : ℕ → P
  | 0 => θ_init
  | n + 1 => U (trajR U D_retain θ_init n) D_retain

/-! ### History-independence -/

/-- A `LocalUpdateRule` is *history-independent* if it ignores its
prior-aggregate argument. -/
def HistoryIndep (U : P → D → P) : Prop :=
  ∀ θ θ' (D : D), U θ D = U θ' D

/-- Causal history-independence with respect to the target client at
initial point `θ_init`: for every round, the retain client's local update
under the original trajectory matches its update under the retrain
trajectory. Matches `theorems_draft.tex` Def. 5 operationally. -/
def CausalHistoryIndep (U : P → D → P)
    (D_target D_retain : D) (w_target w_retain : ℝ) (θ_init : P) : Prop :=
  ∀ n : ℕ,
    U (trajF U D_target D_retain w_target w_retain θ_init n) D_retain
    = U (trajR U D_retain θ_init n) D_retain

/-- Strong HI implies causal HI. -/
lemma HistoryIndep.causalHistoryIndep
    {U : P → D → P} (hU : HistoryIndep U)
    (D_target D_retain : D) (w_target w_retain : ℝ) (θ_init : P) :
    CausalHistoryIndep U D_target D_retain w_target w_retain θ_init := by
  intro n
  rw [hU (trajF U D_target D_retain w_target w_retain θ_init n) θ_init D_retain,
      hU (trajR U D_retain θ_init n) θ_init D_retain]

/-! ### Main theorem -/

/-- **Multi-round Theorem 1 (module-polymorphic, causal-HI form)**: under
FedAvg-2 and *causal* history-independence, gradient subtraction at round
`n + 1` recovers the retrain trajectory at round `n + 1`.

The scalar version (`P = ℝ`) and the multi-dim version (`P = d → ℝ`)
are both special cases. -/
theorem gradUnlearn_fedAvg2_multiRound_causalHI
    (U : P → D → P) (D_target D_retain : D)
    (w_target w_retain : ℝ) (θ_init : P)
    (hCHI : CausalHistoryIndep U D_target D_retain w_target w_retain θ_init)
    (hw_retain : w_retain ≠ 0) (hW : w_target + w_retain ≠ 0)
    (n : ℕ) :
    gradUnlearn_mod
      (trajF U D_target D_retain w_target w_retain θ_init (n + 1))
      (U (trajF U D_target D_retain w_target w_retain θ_init n) D_target)
      w_target (w_target + w_retain)
    = trajR U D_retain θ_init (n + 1) := by
  change gradUnlearn_mod
      (fedAvg2_mod
        (U (trajF U D_target D_retain w_target w_retain θ_init n) D_target)
        (U (trajF U D_target D_retain w_target w_retain θ_init n) D_retain)
        w_target w_retain)
      (U (trajF U D_target D_retain w_target w_retain θ_init n) D_target)
      w_target (w_target + w_retain)
    = trajR U D_retain θ_init (n + 1)
  rw [gradUnlearn_fedAvg2_mod
        (U (trajF U D_target D_retain w_target w_retain θ_init n) D_target)
        (U (trajF U D_target D_retain w_target w_retain θ_init n) D_retain)
        w_target w_retain hw_retain hW,
      hCHI n]
  rfl

/-- **Multi-round Theorem 1 (module-polymorphic, strong-HI form)**:
under strong history-independence, the multi-round identity follows from
the causal-HI form via `HistoryIndep.causalHistoryIndep`. -/
theorem gradUnlearn_fedAvg2_multiRound
    (U : P → D → P) (hU : HistoryIndep U)
    (D_target D_retain : D) (w_target w_retain : ℝ)
    (hw_retain : w_retain ≠ 0) (hW : w_target + w_retain ≠ 0)
    (θ_init : P) (n : ℕ) :
    gradUnlearn_mod
      (trajF U D_target D_retain w_target w_retain θ_init (n + 1))
      (U (trajF U D_target D_retain w_target w_retain θ_init n) D_target)
      w_target (w_target + w_retain)
    = trajR U D_retain θ_init (n + 1) :=
  gradUnlearn_fedAvg2_multiRound_causalHI U D_target D_retain w_target w_retain θ_init
    (hU.causalHistoryIndep D_target D_retain w_target w_retain θ_init) hw_retain hW n

end MultiRoundMD

end FLUL
