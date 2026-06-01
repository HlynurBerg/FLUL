/-
# Federated Learning Types (L.1)

Basic types for the FLUL formalization. See `report/theorems_draft.tex`
Section 1 (Setup and Notation) for the mathematical setup.

This is L.1 of the Lean track — types only. Theorems live in
`Correctness.lean` (L.2), `Characterization.lean` (L.3), and
`Impossibility.lean` (L.4).
-/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace FLUL

/-- A federated learning parameter vector. We work in the scalar (`d = 1`)
case throughout; the multi-dimensional generalization is an open item in
`theorems_draft.tex` Section 6. -/
abbrev Parameters : Type := ℝ

/-- A client identifier. -/
abbrev ClientId : Type := ℕ

/-- A client's local dataset, summarized for scalar (`d = 1`) ridge regression
by the pair `(α, β)` where `α = XᵀX / n` and `β = Xᵀy / n`. Positivity of
`α` is a separate hypothesis at theorem-statement time. -/
structure Dataset where
  /-- Quadratic coefficient `α = XᵀX / n`. -/
  α : ℝ
  /-- Linear coefficient `β = Xᵀy / n`. -/
  β : ℝ

/-- The weight of a client (typically its sample count). Real-valued for
consistency with the algebraic identity in `theorems_draft.tex`. -/
abbrev Weight : Type := ℝ

/-- A federation member: client identity, dataset, and weight. -/
structure Client where
  /-- Stable identifier of this client. -/
  id : ClientId
  /-- This client's local dataset. -/
  data : Dataset
  /-- This client's aggregation weight. -/
  weight : Weight

/-- A local update rule maps `(previous global aggregate, dataset)` to this
client's parameter vector for the next round. A *history-independent* rule
is one that ignores the first argument. -/
abbrev LocalUpdateRule : Type := Parameters → Dataset → Parameters

/-- An aggregator maps a list of `(parameter, weight)` pairs to a global
aggregate. -/
abbrev Aggregator : Type := List (Parameters × Weight) → Parameters

/-- The canonical FedAvg aggregator: weight-normalized weighted average.
Marked `noncomputable` because it divides by the total weight, and `ℝ` is
not a computable field in Lean. -/
noncomputable def fedAvgAggregator : Aggregator := fun cs =>
  let W := cs.foldl (fun acc ⟨_, w⟩ => acc + w) 0
  cs.foldl (fun acc ⟨θ, w⟩ => acc + (w / W) * θ) 0

/-- FedAvg specialized to two clients: `(w₁·θ₁ + w₂·θ₂) / (w₁ + w₂)`.

This convenience function is used by Theorem 2's forward direction
(`Characterization.lean`) and Theorem 3a (`Impossibility.lean`). -/
noncomputable def fedAvg2 (θ_target θ_retain : Parameters)
    (w_target w_retain : Weight) : Parameters :=
  (w_target * θ_target + w_retain * θ_retain) / (w_target + w_retain)

/-- A federated learning protocol bundles a local update rule and an
aggregator. -/
structure Protocol where
  /-- The local update rule used by every client (one rule for all clients
  in the current scalar setup; can be generalized to per-client rules later). -/
  localRule : LocalUpdateRule
  /-- The aggregator. -/
  agg : Aggregator

/-- A trajectory is a sequence of global parameter vectors indexed by round. -/
abbrev Trajectory : Type := ℕ → Parameters

/-- The signature of an exact unlearning operator:
`Λ : (θ_global, θ_target, w_target, W_total) → unlearned_global`.
Matches `theorems_draft.tex` Definition 6. -/
abbrev UnlearnOp : Type := Parameters → Parameters → Weight → Weight → Parameters

/-- Gradient-subtraction unlearning operator from `theorems_draft.tex`
(equation immediately before Theorem 1):
`Λ_grad(θ, θ_★, w_★, W) = (W·θ − w_★·θ_★) / (W − w_★)`. Marked
`noncomputable` because it divides by `W − w_target`. -/
noncomputable def gradUnlearn : UnlearnOp := fun θ θ_target w_target W =>
  (W * θ - w_target * θ_target) / (W - w_target)

/-- Sanity check: in the symmetric 2-client setting (`w_★ = 1`, `W = 2`),
gradient subtraction reduces to `2·θ − θ_★`, the form used in Theorem 3a. -/
example (θ θ_target : ℝ) :
    gradUnlearn θ θ_target 1 2 = 2 * θ - θ_target := by
  unfold gradUnlearn
  ring

end FLUL
