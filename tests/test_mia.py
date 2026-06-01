"""Sanity tests for the MIA library, eval sets, and shadow training."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mia import (
    ThresholdResult,
    evaluate_threshold_attack,
    forward_with_ids,
    lira_offline_scores,
    loss_attack_scores,
    modified_entropy_attack_scores,
)
from mia_eval_sets import _build_transforms, build_evaluation_loader
from mia_shadow import ShadowSpec, compute_shadow_signal_matrix, train_shadow_models
from model import SimpleNet
from utils import train_epoch


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# forward_with_ids: must enforce (img, label, sid) batches and eval mode
# ---------------------------------------------------------------------------


class _TwoTupleDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return torch.zeros(1, 28, 28), 0


class TestForwardWithIds:
    def test_rejects_two_tuple_batches(self, device):
        loader = DataLoader(_TwoTupleDataset(), batch_size=2)
        net = SimpleNet().to(device)
        with pytest.raises(ValueError, match="WithSampleIds"):
            forward_with_ids(net, loader, device)

    def test_sample_id_alignment(self, device):
        ids = [10, 50, 100, 9999]
        loader = build_evaluation_loader("MNIST", ids, train_split=True, batch_size=2)
        net = SimpleNet().to(device)
        data = forward_with_ids(net, loader, device)
        assert set(data.keys()) == set(ids)
        for fields in data.values():
            for k in ("loss", "scaled_logit", "mod_entropy", "label", "prob_correct"):
                assert k in fields, f"missing field {k}"
                assert isinstance(fields[k], (float, int))

    def test_eval_mode_enforced(self, device):
        """SimpleNet has BatchNorm + Dropout — train-mode forwards leak randomness.
        forward_with_ids must call net.eval() so consecutive calls match."""
        ids = [0, 1, 2, 3, 4]
        loader = build_evaluation_loader("MNIST", ids, train_split=True, batch_size=2)
        net = SimpleNet().to(device)
        net.train()
        data1 = forward_with_ids(net, loader, device)
        net.train()
        data2 = forward_with_ids(net, loader, device)
        for sid in ids:
            assert data1[sid]["loss"] == pytest.approx(data2[sid]["loss"], abs=1e-6)
            assert data1[sid]["scaled_logit"] == pytest.approx(
                data2[sid]["scaled_logit"], abs=1e-6
            )


# ---------------------------------------------------------------------------
# Score functions: contract & alignment
# ---------------------------------------------------------------------------


def _per_sample():
    return {
        10: {"loss": 0.1, "scaled_logit": 5.0, "mod_entropy": 0.1, "label": 0, "prob_correct": 0.99},
        5:  {"loss": 0.5, "scaled_logit": 2.0, "mod_entropy": 0.5, "label": 1, "prob_correct": 0.85},
        20: {"loss": 1.0, "scaled_logit": 0.5, "mod_entropy": 1.0, "label": 2, "prob_correct": 0.6},
    }


class TestScoreFunctions:
    def test_loss_scores_sorted_by_sid(self):
        ids, scores = loss_attack_scores(_per_sample())
        assert list(ids) == [5, 10, 20]
        # Higher score = more likely member; we negate loss
        assert scores[1] > scores[0]  # sid=10 has lowest loss
        assert scores[1] > scores[2]

    def test_modent_alignment_matches_loss(self):
        psd = _per_sample()
        ids_l, scores_l = loss_attack_scores(psd)
        ids_m, scores_m = modified_entropy_attack_scores(psd)
        np.testing.assert_array_equal(ids_l, ids_m)
        assert scores_l.shape == scores_m.shape


# ---------------------------------------------------------------------------
# evaluate_threshold_attack
# ---------------------------------------------------------------------------


class TestEvaluateThresholdAttack:
    def test_perfect_separation(self):
        m = np.array([1.0, 1.5, 2.0, 2.5])
        nm = np.array([-1.0, -0.5, 0.0, 0.5])
        result = evaluate_threshold_attack(m, nm)
        assert isinstance(result, ThresholdResult)
        assert result.auc == 1.0
        assert result.balanced_accuracy == 1.0
        assert result.n_members == 4
        assert result.n_nonmembers == 4

    def test_random_scores_near_half(self):
        rng = np.random.default_rng(0)
        m = rng.normal(size=500)
        nm = rng.normal(size=500)
        result = evaluate_threshold_attack(m, nm)
        assert 0.4 < result.auc < 0.6

    def test_drops_nonfinite(self):
        m = np.array([1.0, np.nan, 2.0, np.inf])
        nm = np.array([-1.0, -2.0])
        result = evaluate_threshold_attack(m, nm)
        assert result.n_members == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            evaluate_threshold_attack(np.array([np.nan]), np.array([1.0]))


# ---------------------------------------------------------------------------
# Overfit MIA: a strongly-overfit model must leak through loss/modent attacks
# ---------------------------------------------------------------------------


class TestOverfitMIA:
    @pytest.fixture(scope="class")
    def overfit_scores(self, device):
        torch.manual_seed(42)
        np.random.seed(42)
        member_ids = list(range(50))
        nonmember_ids = list(range(200))
        train_loader = build_evaluation_loader(
            "MNIST", member_ids, train_split=True, batch_size=16
        )
        net = SimpleNet().to(device)
        train_epoch(net, train_loader, device, epochs=30, label_smoothing=0.0)

        m_loader = build_evaluation_loader(
            "MNIST", member_ids, train_split=True, batch_size=64
        )
        nm_loader = build_evaluation_loader(
            "MNIST", nonmember_ids, train_split=False, batch_size=64
        )
        return (
            forward_with_ids(net, m_loader, device),
            forward_with_ids(net, nm_loader, device),
        )

    def test_loss_attack_auc_high(self, overfit_scores):
        m_data, nm_data = overfit_scores
        _, m = loss_attack_scores(m_data)
        _, nm = loss_attack_scores(nm_data)
        result = evaluate_threshold_attack(m, nm)
        assert result.auc > 0.7, f"loss-attack AUC too low: {result.auc:.3f}"

    def test_modent_attack_auc_high(self, overfit_scores):
        m_data, nm_data = overfit_scores
        _, m = modified_entropy_attack_scores(m_data)
        _, nm = modified_entropy_attack_scores(nm_data)
        result = evaluate_threshold_attack(m, nm)
        assert result.auc > 0.7, f"modent-attack AUC too low: {result.auc:.3f}"


# ---------------------------------------------------------------------------
# Transform parity — biggest silent-failure risk for MIA correctness
# ---------------------------------------------------------------------------


class TestTransformParity:
    def test_mnist_transform_matches_utils(self):
        from utils import load_data
        loaders, _ = load_data("MNIST", num_clients=2, batch_size=4)
        # WithSampleIds -> Subset -> raw torchvision dataset
        utils_t = loaders[0].dataset.inner.dataset.transform
        mia_t = _build_transforms("MNIST")
        assert repr(utils_t) == repr(mia_t)


# ---------------------------------------------------------------------------
# Shadow training (uses tmp_path_factory for isolation)
# ---------------------------------------------------------------------------


class TestShadowTraining:
    @pytest.fixture(scope="class")
    def shadow_results(self, tmp_path_factory, device):
        cache = tmp_path_factory.mktemp("shadow_cache")
        spec = ShadowSpec(
            dataset="MNIST",
            n_shadow=2,
            sampling_rate=0.05,
            epochs=1,
            batch_size=128,
            seed=42,
        )
        results = train_shadow_models(spec, cache, device)
        return results, cache, spec

    def test_n_shadows_returned(self, shadow_results):
        results, _, _ = shadow_results
        assert len(results) == 2

    def test_in_sets_differ(self, shadow_results):
        results, _, _ = shadow_results
        assert not np.array_equal(results[0].member_mask, results[1].member_mask)

    def test_in_set_size_matches_sampling_rate(self, shadow_results):
        results, _, _ = shadow_results
        for r in results:
            assert int(r.member_mask.sum()) == 3000  # 60000 * 0.05

    def test_params_differ_between_shadows(self, shadow_results):
        results, _, _ = shadow_results
        l2 = sum(
            float(np.sum((p0 - p1) ** 2))
            for p0, p1 in zip(results[0].params, results[1].params)
        )
        assert l2 > 0.0, "shadows trained on different in-sets should differ"

    def test_cache_hit_returns_identical(self, shadow_results, device):
        results, cache, spec = shadow_results
        results2 = train_shadow_models(spec, cache, device)
        for r1, r2 in zip(results, results2):
            np.testing.assert_array_equal(r1.member_mask, r2.member_mask)
            np.testing.assert_allclose(r1.scaled_logits_full, r2.scaled_logits_full)


# ---------------------------------------------------------------------------
# LiRA offline scoring
# ---------------------------------------------------------------------------


class TestLiRAOffline:
    def test_returns_finite_scores(self):
        rng = np.random.default_rng(0)
        target = {sid: float(rng.normal()) for sid in range(5)}
        shadows = {sid: list(rng.normal(size=5)) for sid in range(5)}
        ids, scores = lira_offline_scores(target, shadows, fixed_variance=True, min_out=2)
        assert len(ids) == 5
        assert np.all(np.isfinite(scores))

    def test_filters_below_min_out(self):
        target = {0: 1.0, 1: 1.0, 2: 1.0}
        shadows = {0: [0.5, 0.6, 0.7], 1: [0.5], 2: []}
        ids, scores = lira_offline_scores(target, shadows, min_out=2)
        assert list(ids) == [0]
        assert scores.shape == (1,)

    def test_higher_z_for_signal_above_distribution(self):
        target = {0: 5.0, 1: -5.0}
        shadows = {
            0: [0.0, 0.1, -0.1, 0.05, -0.05],
            1: [0.0, 0.1, -0.1, 0.05, -0.05],
        }
        ids, scores = lira_offline_scores(target, shadows, min_out=2)
        score_map = dict(zip(ids.tolist(), scores.tolist()))
        assert score_map[0] > score_map[1]


# ---------------------------------------------------------------------------
# compute_shadow_signal_matrix invariant
# ---------------------------------------------------------------------------


def test_signal_matrix_in_plus_out_equals_n_shadow(device, tmp_path):
    spec = ShadowSpec(
        dataset="MNIST",
        n_shadow=2,
        sampling_rate=0.05,
        epochs=1,
        batch_size=128,
        seed=42,
    )
    results = train_shadow_models(spec, tmp_path, device)
    eval_ids = [0, 100, 1000, 5000]
    sig_mat = compute_shadow_signal_matrix(results, eval_ids)
    assert set(sig_mat.keys()) == set(eval_ids)
    for sid in eval_ids:
        in_count = sum(int(r.member_mask[sid]) for r in results)
        out_count = len(sig_mat[sid])
        assert in_count + out_count == len(results)
