/-
# Multi-Round Multi-Dim Theorem 3a (L.4-MR-MD)

Generalisation of `fedProx2_gradUnlearn_residual_md`
(`Theory.ImpossibilityMD`, T = 2) to arbitrary rounds `n ≥ 1`. Mirrors
`Theory.CorrectnessMRMD`'s relationship to `Theory.Correctness`: where
the correctness side carries the multi-round identity through induction
on a history-independent local rule, the impossibility side carries the
*residual* identity through the recursion of the closed-form FedProx
solver in `ℝ^d`.

## Statement

Setup. Each client `c ∈ {target, retain}` holds a Gram matrix
`α_c : Matrix d d ℝ` and a data vector `β_c : d → ℝ`. With FedProx
proximal coefficient `μ`, equal weights, FedAvg aggregator, `θ⁰ = 0`,
the federated trajectory `trajF` and retrain trajectory `trajR`
satisfy the recurrence
* `trajF (n+1) = ½ • ((α_t + μ I)⁻¹·(β_t + μ·trajF n)
                    + (α_r + μ I)⁻¹·(β_r + μ·trajF n))`,
* `trajR (n+1) = (α_r + μ I)⁻¹·(β_r + μ·trajR n)`.

Main result (`fedProx2_gradUnlearn_residual_mr_md`): at round `n + 1`,

    (2 • trajF (n+1) - target_local) - trajR (n+1)
      = μ • (α_r + μ I)⁻¹·(trajF n - trajR n)

where `target_local := (α_t + μ I)⁻¹·(β_t + μ·trajF n)` is the target
client's local update at round `n + 1` along the federated trajectory.

The identity is purely algebraic — uses only linearity of
`Matrix.mulVec`, the FedAvg-2 averaging identity at each round, and
the FedProx step's affine structure. No invertibility hypotheses
required.

## Specialisation to T = 2

At `n = 1`, `trajF 1 - trajR 1 = ½ • (M_t⁻¹·β_t - M_r⁻¹·β_r)`, so the
residual reduces to `(μ/2) • M_r⁻¹·(θ_t¹ - θ_r¹)` — the existing
`Theory.ImpossibilityMD.fedProx2_gradUnlearn_residual_md` exactly. This
is checked as `example` `mrmd_specialises_to_md_at_n_one`.

## Non-vanishing

`fedProx2_gradUnlearn_residual_mr_md_nonzero`: under
`Invertible (α_r + μ • 1)`, `μ > 0`, and a trajectory-disagreement
hypothesis `trajF n ≠ trajR n`, the residual at round `n + 1` is
strictly nonzero. The trajectory disagreement is taken as a hypothesis
because, for general `n`, it does not factor cleanly through a single
round-1 disagreement (the recurrence couples `M_t⁻¹` and `M_r⁻¹` at
each step).

PSD-bridge corollary
`fedProx2_gradUnlearn_residual_mr_md_nonzero_of_posSemidef` discharges
the `Invertible` hypothesis from `α_r.PosSemidef + μ > 0` using the
`invertibleOfPosSemidefAddSmulOne` machinery from
`Theory.ImpossibilityMD`.
-/

import Theory.ImpossibilityMD
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

namespace FLUL

namespace MultiRoundMDImp

variable {d : Type*} [Fintype d] [DecidableEq d]

/-! ### Closed-form FedProx local step and trajectories -/

/-- Closed-form FedProx local solver: `(α + μ I)⁻¹·(β + μ θ_prev)`. -/
noncomputable def fpStep
    (α : Matrix d d ℝ) (β : d → ℝ) (μ : ℝ) (θ_prev : d → ℝ) : d → ℝ :=
  (α + μ • (1 : Matrix d d ℝ))⁻¹.mulVec (β + μ • θ_prev)

/-- Federated 2-client FedProx trajectory with equal weights and
`θ⁰ = 0`. -/
noncomputable def trajF
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ) : ℕ → d → ℝ
  | 0 => 0
  | n + 1 =>
      let prev := trajF α_t α_r β_t β_r μ n
      (1/2 : ℝ) • (fpStep α_t β_t μ prev + fpStep α_r β_r μ prev)

/-- Retrain trajectory: retain client only, `θ⁰ = 0`. -/
noncomputable def trajR
    (α_r : Matrix d d ℝ) (β_r : d → ℝ) (μ : ℝ) : ℕ → d → ℝ
  | 0 => 0
  | n + 1 => fpStep α_r β_r μ (trajR α_r β_r μ n)

/-! ### Helper lemmas -/

/-- The FedAvg-2 averaging identity at every round:
`2 • trajF (n+1) - target_local = retain_local`,
where `target_local`, `retain_local` are the two clients' FedProx
local updates at round `n + 1` along the federated trajectory. -/
lemma two_smul_trajF_succ_sub_target
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ) (n : ℕ) :
    let prev := trajF α_t α_r β_t β_r μ n
    (2 : ℝ) • trajF α_t α_r β_t β_r μ (n + 1) - fpStep α_t β_t μ prev
      = fpStep α_r β_r μ prev := by
  intro prev
  change (2 : ℝ) • ((1/2 : ℝ) • (fpStep α_t β_t μ prev + fpStep α_r β_r μ prev))
        - fpStep α_t β_t μ prev = fpStep α_r β_r μ prev
  ext i
  simp [Pi.sub_apply, Pi.add_apply]

/-- The "shift-by-one-round" residual identity for the retain client:
`fpStep α_r β_r μ trajF_n − trajR (n+1) = μ • M_r⁻¹·(trajF n − trajR n)`,
where `M_r := α_r + μ • 1`. This is the substantive algebraic step;
combined with `two_smul_trajF_succ_sub_target` it gives the main theorem. -/
lemma retain_step_minus_trajR_succ
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ) (n : ℕ) :
    fpStep α_r β_r μ (trajF α_t α_r β_t β_r μ n) -
      trajR α_r β_r μ (n + 1)
      = μ • (α_r + μ • (1 : Matrix d d ℝ))⁻¹.mulVec
            (trajF α_t α_r β_t β_r μ n - trajR α_r β_r μ n) := by
  simp only [fpStep, trajR]
  rw [← Matrix.mulVec_sub]
  have hsub :
      (β_r + μ • trajF α_t α_r β_t β_r μ n) -
        (β_r + μ • trajR α_r β_r μ n)
      = μ • (trajF α_t α_r β_t β_r μ n - trajR α_r β_r μ n) := by
    ext i
    simp [Pi.sub_apply, Pi.add_apply]
    ring
  rw [hsub, Matrix.mulVec_smul]

/-! ### Main theorem -/

/-- **Multi-round multi-dim Theorem 3a, residual identity.**

At round `n + 1` (any `n : ℕ`), the gradient-subtraction unlearning
residual equals `μ • (α_r + μ I)⁻¹` applied to the round-`n`
trajectory disagreement `trajF n − trajR n`:

    (2 • trajF (n+1) − target_local) − trajR (n+1)
      = μ • (α_r + μ I)⁻¹·(trajF n − trajR n).

* At `n = 0`: `trajF 0 = trajR 0 = 0`, so the residual vanishes —
  consistent with gradient subtraction being exact at round 1
  (Theorem 1, algebraic core).
* At `n = 1`: `trajF 1 − trajR 1 = ½ • (M_t⁻¹·β_t − M_r⁻¹·β_r)`, so
  the residual reduces to the existing T = 2 result
  `fedProx2_gradUnlearn_residual_md` exactly.

The identity is purely algebraic — uses only `Matrix.mulVec` linearity,
the FedAvg-2 averaging identity at each round, and the FedProx step's
affine structure. No invertibility, no `μ > 0` hypothesis. -/
theorem fedProx2_gradUnlearn_residual_mr_md
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ) (n : ℕ) :
    let M_r : Matrix d d ℝ := α_r + μ • (1 : Matrix d d ℝ)
    let prev := trajF α_t α_r β_t β_r μ n
    let target_local := fpStep α_t β_t μ prev
    ((2 : ℝ) • trajF α_t α_r β_t β_r μ (n + 1) - target_local)
        - trajR α_r β_r μ (n + 1)
      = μ • M_r⁻¹.mulVec (trajF α_t α_r β_t β_r μ n -
                          trajR α_r β_r μ n) := by
  intro M_r prev target_local
  -- Step 1: FedAvg-2 identity at round n + 1.
  rw [two_smul_trajF_succ_sub_target α_t α_r β_t β_r μ n]
  -- Step 2: retain-step shift-by-one identity.
  exact retain_step_minus_trajR_succ α_t α_r β_t β_r μ n

/-! ### Cross-check: specialises to multi-dim T = 2 at `n = 1`

The existing `fedProx2_gradUnlearn_residual_md` in `Theory.ImpossibilityMD`
proves the residual identity at T = 2 directly. Our multi-round version
at `n = 1` should give the same conclusion (modulo the `½` carried inside
`trajF 1 - trajR 1`). The cross-check below verifies the two formulas
agree on `trajF 1 - trajR 1 = ½ • (M_t⁻¹·β_t - M_r⁻¹·β_r)`. -/

private lemma trajF_one
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ) :
    trajF α_t α_r β_t β_r μ 1
      = (1/2 : ℝ) • ((α_t + μ • (1 : Matrix d d ℝ))⁻¹.mulVec β_t +
                     (α_r + μ • (1 : Matrix d d ℝ))⁻¹.mulVec β_r) := by
  simp only [trajF, fpStep]
  congr 1
  ext i
  simp [Pi.add_apply]

private lemma trajR_one
    (α_r : Matrix d d ℝ) (β_r : d → ℝ) (μ : ℝ) :
    trajR α_r β_r μ 1 = (α_r + μ • (1 : Matrix d d ℝ))⁻¹.mulVec β_r := by
  simp only [trajR, fpStep]
  congr 1
  ext i
  simp [Pi.add_apply]

/-- Cross-check: the multi-round multi-dim residual at `n = 1` matches the
T = 2 identity from `Theory.ImpossibilityMD` (their LHS and RHS agree). -/
example (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ) :
    let M_t : Matrix d d ℝ := α_t + μ • (1 : Matrix d d ℝ)
    let M_r : Matrix d d ℝ := α_r + μ • (1 : Matrix d d ℝ)
    μ • M_r⁻¹.mulVec (trajF α_t α_r β_t β_r μ 1 - trajR α_r β_r μ 1)
      = (μ / 2) • M_r⁻¹.mulVec (M_t⁻¹.mulVec β_t - M_r⁻¹.mulVec β_r) := by
  intro M_t M_r
  rw [trajF_one α_t α_r β_t β_r μ, trajR_one α_r β_r μ]
  -- Goal: μ • M_r⁻¹.mulVec (½ • (M_t⁻¹·β_t + M_r⁻¹·β_r) - M_r⁻¹·β_r)
  --     = (μ/2) • M_r⁻¹.mulVec (M_t⁻¹·β_t - M_r⁻¹·β_r)
  have h_inner :
      (1/2 : ℝ) • (M_t⁻¹.mulVec β_t + M_r⁻¹.mulVec β_r) - M_r⁻¹.mulVec β_r
        = (1/2 : ℝ) • (M_t⁻¹.mulVec β_t - M_r⁻¹.mulVec β_r) := by
    ext i
    simp [Pi.sub_apply, Pi.add_apply, Pi.smul_apply]
    ring
  rw [h_inner, Matrix.mulVec_smul, smul_smul]
  congr 1
  ring

/-! ### Non-vanishing form

The pure algebraic identity above does not say the residual is nonzero.
Under invertibility of `M_r := α_r + μ • 1`, `μ > 0`, and the trajectory
disagreement `trajF n ≠ trajR n`, the residual at round `n + 1` is
genuinely nonzero. Mirrors the T = 2 non-vanishing form. -/

/-- For invertible `M`, the inverse's `mulVec` action has trivial kernel:
`M⁻¹.mulVec v = 0 ↔ v = 0`. Re-proved locally to keep this file's
dependency surface minimal (an identical private lemma lives in
`Theory.ImpossibilityMD`). -/
private lemma mulVec_inv_eq_zero_iff
    (M : Matrix d d ℝ) [Invertible M] (v : d → ℝ) :
    M⁻¹.mulVec v = 0 ↔ v = 0 := by
  refine ⟨fun h => ?_, fun h => by rw [h]; exact Matrix.mulVec_zero _⟩
  have h1 : M.mulVec (M⁻¹.mulVec v) = 0 := by rw [h, Matrix.mulVec_zero]
  rwa [Matrix.mulVec_mulVec, Matrix.mul_inv_of_invertible,
       Matrix.one_mulVec] at h1

/-- **Multi-round multi-dim Theorem 3a, non-vanishing form**: when
`M_r := α_r + μ • 1` is invertible, `μ > 0`, and the round-`n`
trajectory disagreement `trajF n ≠ trajR n`, the gradient-subtraction
residual at round `n + 1` is strictly nonzero.

Mirrors `Theory.ImpossibilityMD.fedProx2_gradUnlearn_residual_md_nonzero`
(the T = 2 case). The trajectory-disagreement hypothesis replaces the
round-1 client disagreement; for general `n` they do not factor cleanly
through a single round-1 quantity. -/
theorem fedProx2_gradUnlearn_residual_mr_md_nonzero
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ)
    (hμ : 0 < μ)
    [Invertible (α_r + μ • (1 : Matrix d d ℝ))]
    (n : ℕ)
    (hΔ : trajF α_t α_r β_t β_r μ n ≠ trajR α_r β_r μ n) :
    let prev := trajF α_t α_r β_t β_r μ n
    let target_local := fpStep α_t β_t μ prev
    ((2 : ℝ) • trajF α_t α_r β_t β_r μ (n + 1) - target_local)
        - trajR α_r β_r μ (n + 1) ≠ 0 := by
  intro prev target_local
  rw [fedProx2_gradUnlearn_residual_mr_md α_t α_r β_t β_r μ n]
  refine smul_ne_zero (ne_of_gt hμ) ?_
  intro h_zero
  rw [mulVec_inv_eq_zero_iff] at h_zero
  exact hΔ (sub_eq_zero.mp h_zero)

/-- **Multi-round multi-dim Theorem 3a, non-vanishing under PSD**: the
natural ridge-regression restatement of the non-vanishing theorem,
using the PSD bridge from `Theory.ImpossibilityMD` to discharge the
`Invertible` instance. -/
theorem fedProx2_gradUnlearn_residual_mr_md_nonzero_of_posSemidef
    (α_t α_r : Matrix d d ℝ) (β_t β_r : d → ℝ) (μ : ℝ)
    (hμ : 0 < μ)
    (hα_r : α_r.PosSemidef)
    (n : ℕ)
    (hΔ : trajF α_t α_r β_t β_r μ n ≠ trajR α_r β_r μ n) :
    let prev := trajF α_t α_r β_t β_r μ n
    let target_local := fpStep α_t β_t μ prev
    ((2 : ℝ) • trajF α_t α_r β_t β_r μ (n + 1) - target_local)
        - trajR α_r β_r μ (n + 1) ≠ 0 := by
  haveI : Invertible (α_r + μ • (1 : Matrix d d ℝ)) :=
    invertibleOfPosSemidefAddSmulOne hα_r hμ
  exact fedProx2_gradUnlearn_residual_mr_md_nonzero
    α_t α_r β_t β_r μ hμ n hΔ

end MultiRoundMDImp

end FLUL
