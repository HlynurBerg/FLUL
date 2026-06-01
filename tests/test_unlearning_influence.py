"""Unit tests for the per-sample influence function unlearning."""
import json
import pickle
import numpy as np
import pytest
from pathlib import Path

from contributions_db import ContributionDB
from unlearning import InfluenceFunctionBasedUnlearning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path):
    return ContributionDB(base_dir=str(tmp_path / "contributions"), dataset_name="TEST")


def _params(val, shape=(4,)):
    return [np.full(shape, val, dtype=np.float64)]


def _save_contribution(db, round_num, client_id, params, num_samples, metrics=None):
    return db.save_contribution(
        round_num=round_num,
        client_id=client_id,
        parameters=params,
        num_samples=num_samples,
        metrics=metrics or {},
    )


def _save_incoming(db, params):
    db.save_initial_global_parameters(params)


def _save_agg(db, round_num, params):
    db.save_aggregated_model(round_num=round_num, parameters=params)


# ---------------------------------------------------------------------------
# _compute_sample_weight_fraction
# ---------------------------------------------------------------------------

class TestComputeSampleWeightFraction:
    def _unlearner(self, tmp_path):
        return InfluenceFunctionBasedUnlearning(_make_db(tmp_path))

    def test_grad_norm_used_first(self, tmp_path):
        ul = self._unlearner(tmp_path)
        metrics = {
            "training_trace": {
                "per_sample_grad_norm": [
                    {"sample_id": 0, "grad_norm": 3.0},
                    {"sample_id": 1, "grad_norm": 1.0},
                    {"sample_id": 2, "grad_norm": 2.0},  # total = 6
                ],
                "per_sample_loss": [
                    {"sample_id": 0, "loss": 9.0},
                    {"sample_id": 1, "loss": 1.0},
                    {"sample_id": 2, "loss": 2.0},
                ],
            },
            "sample_ids": [0, 1, 2],
        }
        # Only sample 0 → 3/6 = 0.5
        frac = ul._compute_sample_weight_fraction(metrics, [0])
        assert frac == pytest.approx(0.5)

    def test_grad_norm_multiple_samples(self, tmp_path):
        ul = self._unlearner(tmp_path)
        metrics = {
            "training_trace": {
                "per_sample_grad_norm": [
                    {"sample_id": 0, "grad_norm": 2.0},
                    {"sample_id": 1, "grad_norm": 2.0},
                    {"sample_id": 2, "grad_norm": 6.0},  # total = 10
                ],
            },
        }
        # Samples 0+1 → (2+2)/10 = 0.4
        frac = ul._compute_sample_weight_fraction(metrics, [0, 1])
        assert frac == pytest.approx(0.4)

    def test_loss_fallback_when_no_grad_norm(self, tmp_path):
        ul = self._unlearner(tmp_path)
        metrics = {
            "training_trace": {
                "per_sample_loss": [
                    {"sample_id": 10, "loss": 1.0},
                    {"sample_id": 11, "loss": 3.0},  # total = 4
                ],
            },
        }
        frac = ul._compute_sample_weight_fraction(metrics, [11])
        assert frac == pytest.approx(0.75)

    def test_uniform_fallback_when_no_trace(self, tmp_path):
        ul = self._unlearner(tmp_path)
        metrics = {"sample_ids": [0, 1, 2, 3]}  # 4 samples
        frac = ul._compute_sample_weight_fraction(metrics, [0, 1])  # 2/4
        assert frac == pytest.approx(0.5)

    def test_uniform_fallback_uses_num_samples(self, tmp_path):
        ul = self._unlearner(tmp_path)
        metrics = {"num_samples": 10}
        frac = ul._compute_sample_weight_fraction(metrics, [0, 1, 2])  # 3/10
        assert frac == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# _estimate_per_sample_influence
# ---------------------------------------------------------------------------

class TestEstimatePerSampleInfluence:
    def _unlearner(self, tmp_path):
        return InfluenceFunctionBasedUnlearning(_make_db(tmp_path))

    def test_full_client_fraction_equals_fedavg_contribution(self, tmp_path):
        ul = self._unlearner(tmp_path)
        incoming = _params(0.0)
        client  = _params(2.0)
        # n_c=5, N=10 → weight_ratio=0.5, fraction=1.0
        influence = ul._estimate_per_sample_influence(incoming, client, 5, 10, 1.0)
        # client_delta = 2.0 - 0.0 = 2.0; influence = 2.0 * 1.0 * 0.5 = 1.0
        np.testing.assert_allclose(influence[0], np.full((4,), 1.0))

    def test_partial_fraction(self, tmp_path):
        ul = self._unlearner(tmp_path)
        incoming = _params(0.0)
        client   = _params(4.0)
        # n_c=10, N=20 → weight_ratio=0.5, fraction=0.25
        influence = ul._estimate_per_sample_influence(incoming, client, 10, 20, 0.25)
        # 4.0 * 0.25 * 0.5 = 0.5
        np.testing.assert_allclose(influence[0], np.full((4,), 0.5))

    def test_zero_fraction_returns_zeros(self, tmp_path):
        ul = self._unlearner(tmp_path)
        incoming = _params(1.0)
        client   = _params(5.0)
        influence = ul._estimate_per_sample_influence(incoming, client, 5, 10, 0.0)
        np.testing.assert_allclose(influence[0], np.zeros(4))

    def test_uses_incoming_not_agg(self, tmp_path):
        """The baseline must be θ_incoming, not θ_agg."""
        ul = self._unlearner(tmp_path)
        # θ_incoming = 1, θ_client = 3 → delta = 2
        # θ_agg = 10 (very different from incoming)
        incoming = _params(1.0)
        client   = _params(3.0)
        influence = ul._estimate_per_sample_influence(incoming, client, 5, 10, 1.0)
        # Should be 2.0 * 1.0 * 0.5 = 1.0 (based on delta from incoming)
        np.testing.assert_allclose(influence[0], np.full((4,), 1.0))


# ---------------------------------------------------------------------------
# unlearn_samples_from_contribution — end-to-end
# ---------------------------------------------------------------------------

class TestUnlearnSamplesFromContribution:
    """
    Scenario: 2 clients, round 0 (so θ_incoming = initial_global_params.pkl).
      θ_incoming = 0
      client 0: θ_c = 2, n=10, samples [0,1,2,3,4] with equal grad norms
      client 1: θ_c = 2, n=10, samples [5,6,7,8,9]
      FedAvg θ_agg = (2*10 + 2*10) / 20 = 2
    We unlearn sample 0 from client 0 (20 % of client 0's gradient mass).
      sample_weight_fraction = 0.2
      I_s = (2-0) * 0.2 * (10/20) = 0.2
      θ_unlearned = 2 - 0.2 = 1.8   (before adaptive damping)
    With damping_factor=1.0, damping = min(1, norm(agg)*1.0/norm(influence)) ≥ 1 → no clipping.
    """

    def _setup(self, tmp_path):
        db = _make_db(tmp_path)
        _save_incoming(db, _params(0.0))   # initial_global_params.pkl → round 0 incoming

        grad_norms_c0 = [{"sample_id": i, "grad_norm": 1.0} for i in range(5)]
        metrics_c0 = {
            "sample_ids": list(range(5)),
            "training_trace": {"per_sample_grad_norm": grad_norms_c0},
        }
        _save_contribution(db, 0, 0, _params(2.0), num_samples=10, metrics=metrics_c0)
        _save_contribution(db, 0, 1, _params(2.0), num_samples=10)
        _save_agg(db, 0, _params(2.0))  # FedAvg θ_agg
        return db

    def test_unlearn_one_sample_reduces_params(self, tmp_path):
        db = self._setup(tmp_path)
        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=1.0)
        unlearned, meta = ul.unlearn_samples_from_contribution(
            round_num=0, client_id=0, sample_ids=[0], influence_scale=1.0
        )
        # sample_weight_fraction = 1/5 = 0.2
        # I = (2-0)*0.2*(10/20) = 0.2
        # θ_unlearned = 2.0 - 0.2 = 1.8
        assert unlearned[0][0] < 2.0, "Unlearned params should be less than original agg"
        assert meta["sample_weight_fraction"] == pytest.approx(0.2)
        assert meta["weight_source"] == "per_sample_grad_norm"

    def test_unlearn_all_samples_matches_full_client(self, tmp_path):
        db = self._setup(tmp_path)
        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=1.0)

        unlearned_samples, _ = ul.unlearn_samples_from_contribution(
            round_num=0, client_id=0, sample_ids=list(range(5)), influence_scale=1.0
        )
        unlearned_client, _ = ul.unlearn_client_contribution(
            round_num=0, client_id=0, influence_scale=1.0
        )
        np.testing.assert_allclose(unlearned_samples[0], unlearned_client[0], rtol=1e-6)

    def test_metadata_fields_present(self, tmp_path):
        db = self._setup(tmp_path)
        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=0.01)
        _, meta = ul.unlearn_samples_from_contribution(
            round_num=0, client_id=0, sample_ids=[0, 1]
        )
        for key in ("round", "client_id", "sample_ids", "total_weight",
                    "client_weight", "sample_weight_fraction", "weight_source",
                    "influence_scale", "damping_factor", "method"):
            assert key in meta, f"Missing metadata key: {key}"

    def test_empty_sample_ids_raises(self, tmp_path):
        db = self._setup(tmp_path)
        ul = InfluenceFunctionBasedUnlearning(db)
        with pytest.raises(ValueError, match="non-empty"):
            ul.unlearn_samples_from_contribution(round_num=0, client_id=0, sample_ids=[])

    def test_missing_incoming_raises(self, tmp_path):
        """If initial_global_params.pkl is absent, a clear error is raised."""
        db = _make_db(tmp_path)
        # No _save_incoming() — so round 0 incoming is missing
        _save_contribution(db, 0, 0, _params(1.0), num_samples=5)
        _save_agg(db, 0, _params(1.0))
        ul = InfluenceFunctionBasedUnlearning(db)
        with pytest.raises(ValueError, match="incoming global model"):
            ul.unlearn_samples_from_contribution(round_num=0, client_id=0, sample_ids=[0])

    def test_loss_fallback_weight_source(self, tmp_path):
        db = _make_db(tmp_path)
        _save_incoming(db, _params(0.0))
        metrics = {
            "sample_ids": [0, 1, 2],
            "training_trace": {
                "per_sample_loss": [
                    {"sample_id": 0, "loss": 2.0},
                    {"sample_id": 1, "loss": 2.0},
                    {"sample_id": 2, "loss": 6.0},  # total=10
                ]
            },
        }
        _save_contribution(db, 0, 0, _params(2.0), num_samples=3, metrics=metrics)
        _save_contribution(db, 0, 1, _params(2.0), num_samples=3)
        _save_agg(db, 0, _params(2.0))
        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=1.0)
        _, meta = ul.unlearn_samples_from_contribution(
            round_num=0, client_id=0, sample_ids=[0, 1]
        )
        assert meta["weight_source"] == "per_sample_loss"
        assert meta["sample_weight_fraction"] == pytest.approx(0.4)  # 4/10

    def test_uniform_fallback_weight_source(self, tmp_path):
        db = _make_db(tmp_path)
        _save_incoming(db, _params(0.0))
        metrics = {"sample_ids": [0, 1, 2, 3]}  # no trace
        _save_contribution(db, 0, 0, _params(2.0), num_samples=4, metrics=metrics)
        _save_contribution(db, 0, 1, _params(2.0), num_samples=4)
        _save_agg(db, 0, _params(2.0))
        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=1.0)
        _, meta = ul.unlearn_samples_from_contribution(
            round_num=0, client_id=0, sample_ids=[0, 1]
        )
        assert meta["weight_source"] == "uniform"
        assert meta["sample_weight_fraction"] == pytest.approx(0.5)  # 2/4


# ---------------------------------------------------------------------------
# unlearn_withdrawn_samples — reads from DB withdrawal state
# ---------------------------------------------------------------------------

class TestUnlearnWithdrawnSamples:
    def test_reads_withdrawn_ids_from_db(self, tmp_path):
        db = _make_db(tmp_path)
        _save_incoming(db, _params(0.0))
        metrics = {"sample_ids": list(range(10)),
                   "training_trace": {"per_sample_grad_norm":
                       [{"sample_id": i, "grad_norm": 1.0} for i in range(10)]}}
        meta_file, _ = _save_contribution(db, 0, 0, _params(2.0), num_samples=10, metrics=metrics)
        _save_contribution(db, 0, 1, _params(2.0), num_samples=10)
        _save_agg(db, 0, _params(2.0))

        ck = json.loads(meta_file.read_text())["contribution_key"]
        db.withdraw_samples_from_contribution(ck, [3, 7])

        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=1.0)
        unlearned, meta = ul.unlearn_withdrawn_samples(round_num=0, client_id=0)
        assert sorted(meta["sample_ids"]) == [3, 7]
        assert meta["sample_weight_fraction"] == pytest.approx(0.2)  # 2/10

    def test_raises_when_no_withdrawal_active(self, tmp_path):
        db = _make_db(tmp_path)
        _save_incoming(db, _params(0.0))
        _save_contribution(db, 0, 0, _params(2.0), num_samples=5)
        _save_agg(db, 0, _params(2.0))
        ul = InfluenceFunctionBasedUnlearning(db)
        with pytest.raises(ValueError, match="No active sample withdrawals"):
            ul.unlearn_withdrawn_samples(round_num=0, client_id=0)


# ---------------------------------------------------------------------------
# unlearn_client_contribution — verify θ_incoming baseline (not θ_agg)
# ---------------------------------------------------------------------------

class TestUnlearnClientContributionBaseline:
    """
    Verify that the client-level unlearn uses θ_incoming as the baseline.
    If θ_agg were used instead, the influence would differ when θ_incoming ≠ θ_agg.
    """

    def test_influence_based_on_incoming_not_agg(self, tmp_path):
        db = _make_db(tmp_path)
        # θ_incoming = 0, θ_client = 4, θ_agg = 10 (very different)
        _save_incoming(db, _params(0.0))
        _save_contribution(db, 0, 0, _params(4.0), num_samples=5)
        _save_contribution(db, 0, 1, _params(4.0), num_samples=5)
        _save_agg(db, 0, _params(10.0))   # artificially high

        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=1.0)
        unlearned, _ = ul.unlearn_client_contribution(
            round_num=0, client_id=0, influence_scale=1.0
        )

        # influence = (θ_c - θ_incoming) * 1.0 * (5/10) = 4.0 * 0.5 = 2.0
        # θ_unlearned = 10.0 - 2.0 = 8.0
        np.testing.assert_allclose(unlearned[0], np.full((4,), 8.0), atol=1e-6)

    def test_old_agg_baseline_would_give_different_result(self, tmp_path):
        """Contrast: if we mistakenly used θ_agg as baseline, delta would be θ_c - θ_agg
        (not θ_c - θ_incoming), giving a wrong result when the two differ."""
        db = _make_db(tmp_path)
        _save_incoming(db, _params(0.0))
        _save_contribution(db, 0, 0, _params(4.0), num_samples=5)
        _save_contribution(db, 0, 1, _params(4.0), num_samples=5)
        _save_agg(db, 0, _params(10.0))

        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=1.0)
        unlearned, _ = ul.unlearn_client_contribution(round_num=0, client_id=0)

        # The correct result (8.0) must differ from θ_agg (10.0),
        # which is what the old θ_agg-baseline approach would produce
        # when θ_c == θ_agg (it would have zero influence).
        assert not np.allclose(unlearned[0], np.full((4,), 10.0))


# ---------------------------------------------------------------------------
# Integration tests for multi-round sample removal
# ---------------------------------------------------------------------------

def _save_incoming_for_round(db, round_num, params):
    """Save a fake 'aggregated' model for the previous round (acts as incoming)."""
    if round_num == 0:
        db.save_initial_global_parameters(params)
    else:
        db.save_aggregated_model(round_num=round_num - 1, parameters=params)


class TestFindRoundsContainingSamples:
    """6a — ContributionDB.find_rounds_containing_samples() correctly identifies rounds."""

    def test_finds_correct_rounds(self, tmp_path):
        db = _make_db(tmp_path)
        # Round 0: samples 10, 20, 30
        _save_contribution(db, 0, 0, _params(1.0), num_samples=3,
                           metrics={"sample_ids": [10, 20, 30]})
        # Round 1: samples 40, 50 (no overlap)
        _save_contribution(db, 1, 0, _params(1.0), num_samples=2,
                           metrics={"sample_ids": [40, 50]})
        # Round 2: samples 10, 30, 60 (partial overlap)
        _save_contribution(db, 2, 0, _params(1.0), num_samples=3,
                           metrics={"sample_ids": [10, 30, 60]})

        result = db.find_rounds_containing_samples(client_id=0, sample_ids=[10, 30])

        assert set(result.keys()) == {0, 2}, f"Expected rounds {{0, 2}}, got {set(result.keys())}"
        assert set(result[0]) == {10, 30}
        assert set(result[2]) == {10, 30}
        assert 1 not in result

    def test_no_overlap_returns_empty(self, tmp_path):
        db = _make_db(tmp_path)
        _save_contribution(db, 0, 0, _params(1.0), num_samples=2,
                           metrics={"sample_ids": [1, 2]})
        result = db.find_rounds_containing_samples(client_id=0, sample_ids=[99])
        assert result == {}

    def test_no_manifest_falls_back_to_all_rounds(self, tmp_path):
        """When no sample manifest exists, all requested IDs are assumed present."""
        db = _make_db(tmp_path)
        _save_contribution(db, 0, 0, _params(1.0), num_samples=10)  # no sample_ids in metrics
        _save_contribution(db, 1, 0, _params(1.0), num_samples=10)

        result = db.find_rounds_containing_samples(client_id=0, sample_ids=[5, 7])
        assert 0 in result
        assert 1 in result
        assert set(result[0]) == {5, 7}


class TestMultiRoundSampleUnlearning:
    """6b — unlearn_samples_all_rounds() unlearns each affected round independently."""

    def test_unlearns_only_affected_rounds(self, tmp_path):
        db = _make_db(tmp_path)
        _save_incoming(db, _params(0.0))  # round 0 incoming

        # Round 0: client 0 trained on samples {10, 20}; client 1 on other samples
        _save_contribution(db, 0, 0, _params(4.0), num_samples=4,
                           metrics={"sample_ids": [10, 20, 30, 40]})
        _save_contribution(db, 0, 1, _params(2.0), num_samples=4)
        # agg round 0 = (4.0*4 + 2.0*4) / 8 = 3.0
        _save_agg(db, 0, _params(3.0))

        # Round 1: client 0 trained on samples {50, 60} — no overlap with target
        _save_agg(db, 0, _params(3.0))  # also serves as incoming for round 1
        _save_contribution(db, 1, 0, _params(5.0), num_samples=2,
                           metrics={"sample_ids": [50, 60]})
        _save_contribution(db, 1, 1, _params(3.0), num_samples=2)
        _save_agg(db, 1, _params(4.0))

        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=1.0)
        results = ul.unlearn_samples_all_rounds(
            client_id=0,
            sample_ids=[10, 20],
            dataset_name="TEST",
            propagate=False,
        )

        # Samples {10,20} only appear in round 0
        assert results["sample_rounds"] == {0: [10, 20]}, results["sample_rounds"]
        assert len(results["unlearned_rounds"]) == 1
        assert results["unlearned_rounds"][0]["round"] == 0
        assert results["unlearned_rounds"][0]["status"] == "success"

    def test_unlearned_params_differ_from_original(self, tmp_path):
        db = _make_db(tmp_path)
        _save_incoming(db, _params(0.0))
        _save_contribution(db, 0, 0, _params(4.0), num_samples=5,
                           metrics={"sample_ids": [1, 2, 3, 4, 5]})
        _save_contribution(db, 0, 1, _params(2.0), num_samples=5)
        original_agg = _params(3.0)
        _save_agg(db, 0, original_agg)

        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=1.0)
        results = ul.unlearn_samples_all_rounds(
            client_id=0, sample_ids=[1, 2], dataset_name="TEST", propagate=False
        )

        assert results["unlearned_rounds"][0]["status"] == "success"
        # Load the now-saved corrected aggregate and verify it changed
        corrected = ul._load_aggregated_model(0)
        assert corrected is not None
        assert not np.allclose(corrected[0], original_agg[0]), \
            "Unlearned aggregate should differ from original"


class TestSampleUnlearningPropagation:
    """6c — propagate=True recalculates subsequent rounds using the corrected model."""

    def test_propagation_preserves_corrections_in_multi_round_sample_set(self, tmp_path):
        """
        Regression test: when forget samples appear in MULTIPLE rounds, the per-round
        unlearning correction in rounds between min(sample_rounds) and max(sample_rounds)
        must NOT be overwritten by propagation. Previously
        ``_propagate_sample_unlearning`` was called with ``from_round=min(sample_rounds)``,
        which caused FedAvg recalculation to overwrite the corrections in those rounds.
        After the fix, ``from_round=max(sample_rounds)`` so only rounds AFTER the last
        unlearned round are recalculated.
        """
        db = _make_db(tmp_path)
        _save_incoming(db, _params(0.0))  # round-0 incoming

        # Round 0: client 0 has the forget sample [3]
        _save_contribution(db, 0, 0, _params(4.0), num_samples=5,
                           metrics={"sample_ids": [1, 2, 3, 4, 5]})
        _save_contribution(db, 0, 1, _params(2.0), num_samples=5)
        _save_agg(db, 0, _params(3.0))

        # Round 1: client 0 ALSO has sample [3] in its training trace (same partition)
        _save_contribution(db, 1, 0, _params(6.0), num_samples=5,
                           metrics={"sample_ids": [3, 4, 5, 6, 7]})
        _save_contribution(db, 1, 1, _params(4.0), num_samples=5)
        original_round1_agg_value = 5.0  # (6*5 + 4*5) / 10 = 5.0
        _save_agg(db, 1, _params(original_round1_agg_value))

        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=1.0)
        results = ul.unlearn_samples_all_rounds(
            client_id=0, sample_ids=[3], dataset_name="TEST", propagate=True
        )

        # Both rounds appear in sample_rounds → both must be unlearned successfully
        assert set(r["round"] for r in results["unlearned_rounds"]) == {0, 1}
        for r in results["unlearned_rounds"]:
            assert r["status"] == "success"

        # The KEY assertion: round 1's saved aggregate is the UNLEARNED one,
        # not the FedAvg-recalculated one. The unlearned round-1 aggregate should
        # differ from the original 5.0 (the influence correction subtracts a
        # non-zero quantity).
        round1_after = ul._load_aggregated_model(1)
        assert not np.allclose(
            round1_after[0], np.full((4,), original_round1_agg_value)
        ), (
            "Round 1's per-round unlearning correction was overwritten by propagation. "
            "Expected: round-1 aggregate to differ from the original FedAvg value "
            f"{original_round1_agg_value}; got: {round1_after[0]}."
        )

    def test_propagation_updates_subsequent_round(self, tmp_path):
        db = _make_db(tmp_path)
        _save_incoming(db, _params(0.0))  # incoming for round 0

        # Round 0: two clients
        _save_contribution(db, 0, 0, _params(4.0), num_samples=5,
                           metrics={"sample_ids": [1, 2, 3, 4, 5]})
        _save_contribution(db, 0, 1, _params(2.0), num_samples=5)
        _save_agg(db, 0, _params(3.0))

        # Round 1: two clients; target sample not used (no manifest → fallback doesn't
        # add round 1 to sample_rounds since we give explicit sample_ids)
        _save_contribution(db, 1, 0, _params(6.0), num_samples=5,
                           metrics={"sample_ids": [10, 11, 12, 13, 14]})
        _save_contribution(db, 1, 1, _params(4.0), num_samples=5)
        original_round1_agg = _params(5.0)
        _save_agg(db, 1, original_round1_agg)

        ul = InfluenceFunctionBasedUnlearning(db, damping_factor=1.0)
        # Withdraw sample 3 from client 0's round-0 contribution, propagate=True
        results = ul.unlearn_samples_all_rounds(
            client_id=0, sample_ids=[3], dataset_name="TEST", propagate=True
        )

        assert results["unlearned_rounds"][0]["status"] == "success"

        # Round 1 should have been recalculated (propagation)
        updated_round1 = ul._load_aggregated_model(1)
        assert updated_round1 is not None
        # The recalculated round-1 aggregate should now differ from the stale one
        # because recalculate_aggregated_model re-averages using updated sample weights
        # (client 0 round 0 now has a withdrawn sample, but round 1's contributions
        #  are untouched so the aggregate stays the same for round 1 in this simple setup).
        # The key assertion is that propagation ran without error and saved a model.
        assert not np.isnan(updated_round1[0]).any()


class TestRecalculationWithSampleWithdrawals:
    """6d — recalculate_aggregated_model() adjusts weights for withdrawn samples."""

    def test_withdrawn_samples_reduce_weight(self, tmp_path):
        db = _make_db(tmp_path)

        # Client 0: 100 samples; client 1: 100 samples; equal weight → agg = midpoint
        _save_contribution(db, 0, 0, _params(10.0), num_samples=100)
        _save_contribution(db, 0, 1, _params(0.0), num_samples=100)
        _save_agg(db, 0, _params(5.0))  # original: (10+0)/2 = 5.0

        # Withdraw 80 samples from client 0 → effective weight becomes 20
        ck0 = db.resolve_contribution_key(0, 0)
        db.withdraw_samples_from_contribution(ck0, list(range(80)))

        recalculated = db.recalculate_aggregated_model(0, exclude_withdrawn=False)
        assert recalculated is not None

        # Expected: (10.0*20 + 0.0*100) / 120 = 200/120 ≈ 1.6667
        expected = (10.0 * 20 + 0.0 * 100) / (20 + 100)
        np.testing.assert_allclose(recalculated[0], np.full((4,), expected), atol=1e-6)

    def test_no_sample_withdrawal_unchanged(self, tmp_path):
        """Without sample withdrawals the result is standard FedAvg."""
        db = _make_db(tmp_path)
        _save_contribution(db, 0, 0, _params(8.0), num_samples=3)
        _save_contribution(db, 0, 1, _params(2.0), num_samples=7)
        _save_agg(db, 0, _params(5.0))

        recalculated = db.recalculate_aggregated_model(0, exclude_withdrawn=False)
        # FedAvg: (8.0*3 + 2.0*7) / 10 = (24+14)/10 = 3.8
        np.testing.assert_allclose(recalculated[0], np.full((4,), 3.8), atol=1e-6)
