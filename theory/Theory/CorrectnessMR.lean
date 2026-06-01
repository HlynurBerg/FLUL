/-
# Multi-Round Theorem 1 (L.2-MR, scalar specialization of L.2-MR-MD)

Scalar (`Parameters = ℝ`) wrapper around `CorrectnessMRMD.lean`. The
polymorphic theorem `MultiRoundMD.gradUnlearn_fedAvg2_multiRound_causalHI`
covers this case at `P = ℝ`; this module exposes the scalar form with the
scalar `gradUnlearn` / `Dataset` API used by the thesis prose
(`theorems_draft.tex` Sections 4–5, Theorem 1).

The `HistoryIndep` / `CausalHistoryIndep` predicates and the
`MultiRound.step` / `trajF` / `trajR` trajectories are `abbrev`s pointing
to their `MultiRoundMD` counterparts at `P = ℝ` (`Module ℝ ℝ` is
automatic), so propositional equality is automatic and no trajectory-level
bridges are needed. The only real bridge is `gradUnlearn_eq_mod`, since
scalar `gradUnlearn` (from `Theory.FL`, defined via `*` and `/`) differs
syntactically from polymorphic `gradUnlearn_mod` (defined via `•`).
-/

import Theory.FL
import Theory.Correctness
import Theory.CorrectnessMRMD

namespace FLUL

/-! ## Scalar / polymorphic bridge at `P = ℝ`

Only `gradUnlearn` needs an explicit bridge — everything else is an
`abbrev` and reduces automatically. -/

/-- Scalar `gradUnlearn` equals the polymorphic `gradUnlearn_mod` at
`P = ℝ`. Both express `(W·θ − w_t·θ_t) / (W − w_t)`; scalar uses `*` and
`/`, polymorphic uses `•`. -/
lemma gradUnlearn_eq_mod (θ θ_t w_t W : ℝ) :
    gradUnlearn θ θ_t w_t W
    = MultiRoundMD.gradUnlearn_mod (P := ℝ) θ θ_t w_t W := by
  unfold gradUnlearn MultiRoundMD.gradUnlearn_mod
  simp [smul_eq_mul]
  ring

/-! ## Scalar abbrevs — specializations of `MultiRoundMD` at `P = ℝ` -/

/-- A `LocalUpdateRule` is *history-independent* if it ignores its
prior-aggregate argument. -/
abbrev HistoryIndep (U : LocalUpdateRule) : Prop :=
  MultiRoundMD.HistoryIndep (P := ℝ) (D := Dataset) U

namespace MultiRound

/-- One step of the 2-client scalar FedAvg protocol. -/
noncomputable abbrev step (U : LocalUpdateRule) (D_target D_retain : Dataset)
    (w_target w_retain : Weight) (θ_prev : Parameters) : Parameters :=
  MultiRoundMD.step (P := ℝ) U D_target D_retain w_target w_retain θ_prev

/-- Original 2-client FedAvg trajectory. -/
noncomputable abbrev trajF (U : LocalUpdateRule) (D_target D_retain : Dataset)
    (w_target w_retain : Weight) (θ_init : Parameters) : ℕ → Parameters :=
  MultiRoundMD.trajF (P := ℝ) U D_target D_retain w_target w_retain θ_init

/-- Retrain trajectory: only the retain client participates. -/
noncomputable abbrev trajR (U : LocalUpdateRule) (D_retain : Dataset)
    (θ_init : Parameters) : ℕ → Parameters :=
  MultiRoundMD.trajR (P := ℝ) U D_retain θ_init

end MultiRound

open MultiRound

/-- A 2-client FedAvg protocol is *causally history-independent* with
respect to the target client at initial point `θ_init`
(`theorems_draft.tex` Def. 5). -/
abbrev CausalHistoryIndep (U : LocalUpdateRule) (D_target D_retain : Dataset)
    (w_target w_retain : Weight) (θ_init : Parameters) : Prop :=
  MultiRoundMD.CausalHistoryIndep (P := ℝ)
    U D_target D_retain w_target w_retain θ_init

/-- Strong history-independence implies causal history-independence. -/
lemma HistoryIndep.causalHistoryIndep
    {U : LocalUpdateRule} (hU : HistoryIndep U)
    (D_target D_retain : Dataset) (w_target w_retain : Weight)
    (θ_init : Parameters) :
    CausalHistoryIndep U D_target D_retain w_target w_retain θ_init :=
  MultiRoundMD.HistoryIndep.causalHistoryIndep hU
    D_target D_retain w_target w_retain θ_init

/-! ## Multi-round Theorem 1 (scalar form) -/

/-- **Multi-round Theorem 1 (scalar, causal-HI form)**: under FedAvg-2 and
*causal* history-independence (`theorems_draft.tex` Def. 5), gradient
subtraction at round `n + 1` recovers the retrain trajectory at round
`n + 1`.

Proof. The `abbrev` aliases reduce scalar `trajF`/`trajR`/`CausalHistoryIndep`
to their `MultiRoundMD` counterparts at `P = ℝ`. The algebraic bridge
`gradUnlearn_eq_mod` rewrites scalar `gradUnlearn` to polymorphic
`gradUnlearn_mod`. The remaining goal is exactly
`MultiRoundMD.gradUnlearn_fedAvg2_multiRound_causalHI`. -/
theorem gradUnlearn_fedAvg2_multiRound_causalHI
    (U : LocalUpdateRule) (D_target D_retain : Dataset)
    (w_target w_retain : Weight) (θ_init : Parameters)
    (hCHI : CausalHistoryIndep U D_target D_retain w_target w_retain θ_init)
    (hw_retain : w_retain ≠ 0) (hW : w_target + w_retain ≠ 0)
    (n : ℕ) :
    gradUnlearn
      (trajF U D_target D_retain w_target w_retain θ_init (n + 1))
      (U (trajF U D_target D_retain w_target w_retain θ_init n) D_target)
      w_target (w_target + w_retain)
    = trajR U D_retain θ_init (n + 1) := by
  rw [gradUnlearn_eq_mod]
  exact MultiRoundMD.gradUnlearn_fedAvg2_multiRound_causalHI
    U D_target D_retain w_target w_retain θ_init hCHI hw_retain hW n

/-- **Multi-round Theorem 1 (scalar, strong-HI form)**: under strong
history-independence, the multi-round identity follows from the causal-HI
form via `HistoryIndep.causalHistoryIndep`. -/
theorem gradUnlearn_fedAvg2_multiRound
    (U : LocalUpdateRule) (hU : HistoryIndep U)
    (D_target D_retain : Dataset) (w_target w_retain : Weight)
    (hw_retain : w_retain ≠ 0) (hW : w_target + w_retain ≠ 0)
    (θ_init : Parameters) (n : ℕ) :
    gradUnlearn
      (trajF U D_target D_retain w_target w_retain θ_init (n + 1))
      (U (trajF U D_target D_retain w_target w_retain θ_init n) D_target)
      w_target (w_target + w_retain)
    = trajR U D_retain θ_init (n + 1) :=
  gradUnlearn_fedAvg2_multiRound_causalHI U D_target D_retain w_target w_retain θ_init
    (hU.causalHistoryIndep D_target D_retain w_target w_retain θ_init) hw_retain hW n

end FLUL
