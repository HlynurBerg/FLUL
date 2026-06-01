/-
# FLUL Theory — Library Root

Machine-checked formalization of federated unlearning, accompanying
the thesis empirical study in `report/main.tex` and the theorem drafts in
`report/theorems_draft.tex`.

Modules:
* `Theory.FL` (L.1, complete): federated learning types and the
  unlearning-operator signature.
* `Theory.Correctness` (L.2, complete): correctness theorem for gradient
  subtraction in the 2-client FedAvg single-round case (`gradUnlearn_fedAvg2`
  + symmetric corollary).
* `Theory.CorrectnessMR` (L.2-MR): multi-round Theorem 1
  (`gradUnlearn_fedAvg2_multiRound`) under `HistoryIndep` local rules; also
  `gradUnlearn_fedAvg2_multiRound_causalHI` under the operational `Def. 5`
  causal-HI form. Scalar (`Parameters = ℝ`) only.
* `Theory.CorrectnessMRMD` (L.2-MR-MD): module-polymorphic multi-round
  Theorem 1 (`gradUnlearn_fedAvg2_multiRound_causalHI` in
  `MultiRoundMD` namespace). Stated over arbitrary `ℝ`-modules `P`;
  the scalar case (`P = ℝ`) and the multi-dim case (`P = d → ℝ`) are both
  instances. Uses `fedAvg2_mod` / `gradUnlearn_mod` defined via `ℝ`-smul.
* `Theory.Characterization` (L.3, both directions complete): forward
  direction (`characterization_forward_FedAvg2`, witness `gradUnlearn`) is
  complete for FedAvg-2; reverse direction is complete for both branches —
  non-linear aggregator (`characterization_reverse_MedianAgg3`, the
  3-client median counterexample) and history-dependent local rule
  (`characterization_reverse_FedProx2`, the 2-client FedProx
  parametric-in-μ counterexample, mirrors `fp2_paramInstance*` lemmas from
  `Impossibility.lean`).
* `Theory.Impossibility` (L.4 complete, Theorems 3a and 3b):
  - **3a** (`fedProx2_gradUnlearn_residual`): exact closed-form residual for
    gradient subtraction under 2-client T=2 FedProx, plus non-vanishing
    corollary.
  - **3b** (`fedProx2_no_exact_data_indep_linear`): for every `μ > 0`, no
    data-independent linear operator `a · θ² + b · θ_★²` is exact across all
    data choices. Parametric three-instance witness whose constraints
    collapse to a μ-invariant integer linear system.
* `Theory.ImpossibilityMD` (L.4-MD, multi-dim 3a):
  - **3a-MD** (`fedProx2_gradUnlearn_residual_md`): polymorphic generalization
    of Theorem 3a's residual to `ℝ^d`. The scalar `μ/(2(α_r+μ))` becomes the
    matrix factor `(μ/2)·(α_r + μ I)⁻¹` acting on `θ_★¹ - θ_r¹`. Proved via
    linearity of `Matrix.mulVec` — no invertibility hypothesis needed.
  - **3a-MD non-vanishing** (`fedProx2_gradUnlearn_residual_md_nonzero`):
    under `Invertible (α_r + μ•1)`, `μ > 0`, and round-1 disagreement, the
    residual is strictly nonzero. Mirrors scalar
    `fedProx2_gradUnlearn_residual_nonzero`. Uses helper
    `mulVec_inv_eq_zero_iff` for invertible matrices.
  - **PSD bridge** (`smul_one_posDef`, `posSemidef_add_smul_one_posDef`,
    `invertibleOfPosSemidefAddSmulOne`,
    `fedProx2_gradUnlearn_residual_md_nonzero_of_posSemidef`): users with
    the natural ridge-regression hypothesis `α_r.PosSemidef + μ > 0` can
    invoke the non-vanishing theorem directly, with the `Invertible`
    instance discharged via `Matrix.PosDef.posSemidef_add` and
    `Matrix.PosDef.isUnit`.
* `Theory.ImpossibilityMRMD` (L.4-MR-MD, multi-round multi-dim 3a):
  - **3a-MR-MD** (`fedProx2_gradUnlearn_residual_mr_md`): at any round
    `n + 1 ≥ 1`, the gradient-subtraction residual equals
    `μ • (α_r + μ I)⁻¹` applied to the round-`n` trajectory
    disagreement `trajF n − trajR n`. Pure algebraic identity, no
    invertibility hypothesis. Generalises
    `Theory.ImpossibilityMD.fedProx2_gradUnlearn_residual_md` from
    `T = 2` to arbitrary `T = n + 1`; an `example` block verifies the
    two formulas agree on `trajF 1 − trajR 1`.
  - **3a-MR-MD non-vanishing** (`fedProx2_gradUnlearn_residual_mr_md_nonzero`)
    and PSD-bridge corollary
    (`fedProx2_gradUnlearn_residual_mr_md_nonzero_of_posSemidef`): under
    invertibility (or PSD + `μ > 0`) and round-`n` trajectory disagreement
    `trajF n ≠ trajR n`, the residual at round `n + 1` is strictly nonzero.
* `Theory.ImpossibilityMDB` (L.4-MDB, multi-dim 3b):
  - **3b-MD** (`fedProx2_no_exact_data_indep_linear_md`): for every `μ > 0`
    and `i₀ : d`, no *matrix* pair `(A, B) ∈ (Matrix d d ℝ)²` is exact for
    multi-dim FedProx-2-2. Proof reduces to scalar 3b via scalar-embedded
    data `(α • 1, Pi.single i₀ β)` — every multi-dim round-2 quantity at
    such data is `(scalar fp2) • Pi.single i₀ 1`, so projecting on `i₀`
    yields the scalar 3b constraint on `(A i₀ i₀, B i₀ i₀)`.
-/

import Theory.FL
import Theory.Correctness
import Theory.CorrectnessMR
import Theory.CorrectnessMRMD
import Theory.Characterization
import Theory.Impossibility
import Theory.ImpossibilityMD
import Theory.ImpossibilityMRMD
import Theory.ImpossibilityMDB
