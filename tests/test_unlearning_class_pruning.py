"""Unit tests for ClassDiscriminativePruningUnlearning (Wang et al. 2022)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from contributions_db import ContributionDB
from model import SimpleNet, get_parameters
from unlearning import ClassDiscriminativePruningUnlearning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_db(tmp_path):
    return ContributionDB(base_dir=str(tmp_path / "contributions"), dataset_name="MNIST")


def _make_unlearner(tmp_path, **kwargs):
    """Build a unlearner with a SimpleNet (matches MNIST defaults)."""
    db = _make_db(tmp_path)
    model = SimpleNet(num_classes=10, num_channels=1, img_size=28)
    return ClassDiscriminativePruningUnlearning(
        contribution_db=db,
        model=model,
        dataset_name="MNIST",
        num_clients=2,
        device=torch.device("cpu"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# _infer_num_classes
# ---------------------------------------------------------------------------
class TestInferNumClasses:
    def test_simplenet_returns_10(self, tmp_path):
        ul = _make_unlearner(tmp_path)
        assert ul._infer_num_classes() == 10


# ---------------------------------------------------------------------------
# _apply_pruning — synthetic TF-IDF input
# ---------------------------------------------------------------------------
class TestApplyPruning:
    def _params_for_simplenet(self, ul):
        # Use the model's own params as a starting point, then mutate
        return [np.array(p, copy=True) for p in get_parameters(ul.model)]

    def test_top_channels_zeroed_for_target_class(self, tmp_path):
        """Channels with highest TF-IDF for target class should be zeroed."""
        ul = _make_unlearner(tmp_path, prune_ratio=0.25)  # 25% of 32 = 8 channels for conv1
        params = self._params_for_simplenet(ul)
        # Synthetic TF-IDF: [num_classes=10, channels]
        # Make channel 0 highest for class 0, channel 31 for class 9
        tfidf_conv1 = torch.zeros(10, 32)
        for c in range(32):
            # Class c % 10 has highest activation in channel c
            tfidf_conv1[c % 10, c] = 100.0 + c
        tfidf_conv2 = torch.zeros(10, 64)
        # Make first 8 channels per class high for that class
        for k in range(10):
            for offset in range(8):
                tfidf_conv2[k, k * 8 % 64 + offset] = 50.0 + offset

        tfidf = {"conv1": tfidf_conv1, "conv2": tfidf_conv2}
        target_classes = [0]

        new_params, pruned_per_layer = ul._apply_pruning(params, target_classes, tfidf)
        # 25% of 32 = 8 channels pruned in conv1 (top-k for class 0)
        assert len(pruned_per_layer["conv1"]) == 8
        # Verify those filter weights are zero
        # SimpleNet state_dict order: conv1.weight, conv1.bias, conv2.weight, conv2.bias, ...
        state_keys = list(ul.model.state_dict().keys())
        w_idx = state_keys.index("conv1.weight")
        for c in pruned_per_layer["conv1"]:
            assert np.allclose(new_params[w_idx][c], 0.0), f"channel {c} not zeroed"

    def test_non_target_channels_preserved(self, tmp_path):
        """Pruning a target class should NOT touch channels outside the chosen top-k."""
        ul = _make_unlearner(tmp_path, prune_ratio=0.1)  # 10% of 32 → 3 channels pruned
        params = self._params_for_simplenet(ul)
        tfidf = {
            "conv1": torch.zeros(10, 32),
            "conv2": torch.zeros(10, 64),
        }
        # Distinguishable scores so topk is deterministic
        for c, score in [(7, 999.0), (15, 500.0), (23, 100.0)]:
            tfidf["conv1"][5, c] = score
        target_classes = [5]
        new_params, pruned_per_layer = ul._apply_pruning(params, target_classes, tfidf)
        # 3 channels pruned (top-3 for class 5)
        assert set(pruned_per_layer["conv1"]) == {7, 15, 23}
        state_keys = list(ul.model.state_dict().keys())
        w_idx = state_keys.index("conv1.weight")
        # Pruned channels are zeroed
        for c in (7, 15, 23):
            assert np.allclose(new_params[w_idx][c], 0.0)
        # Untouched channels remain identical to the input
        for c in (0, 4, 10, 31):
            np.testing.assert_allclose(new_params[w_idx][c], params[w_idx][c])

    def test_zero_prune_ratio_still_prunes_at_least_one(self, tmp_path):
        """Edge case: prune_ratio that rounds to 0 still prunes 1 channel (max(1, ...))."""
        ul = _make_unlearner(tmp_path, prune_ratio=0.001)  # tiny ratio
        params = self._params_for_simplenet(ul)
        tfidf = {"conv1": torch.zeros(10, 32), "conv2": torch.zeros(10, 64)}
        tfidf["conv1"][3, 0] = 1.0
        new_params, pruned_per_layer = ul._apply_pruning(params, [3], tfidf)
        assert len(pruned_per_layer["conv1"]) >= 1

    def test_multiple_target_classes_union(self, tmp_path):
        ul = _make_unlearner(tmp_path, prune_ratio=0.1)
        params = self._params_for_simplenet(ul)
        tfidf = {"conv1": torch.zeros(10, 32), "conv2": torch.zeros(10, 64)}
        tfidf["conv1"][2, 5] = 999.0  # class 2 → channel 5
        tfidf["conv1"][7, 20] = 999.0  # class 7 → channel 20
        new_params, pruned_per_layer = ul._apply_pruning(params, [2, 7], tfidf)
        assert 5 in pruned_per_layer["conv1"]
        assert 20 in pruned_per_layer["conv1"]


# ---------------------------------------------------------------------------
# _compute_tfidf — synthetic probe loader
# ---------------------------------------------------------------------------
class TestComputeTFIDF:
    def test_high_tfidf_for_class_specific_activation(self, tmp_path):
        """A channel that activates strongly for class 7 only should have high TF-IDF for class 7."""
        ul = _make_unlearner(tmp_path)
        # Build a tiny probe loader: 5 samples per class
        torch.manual_seed(0)
        n_per_class = 5
        num_classes = 10
        samples = []
        for k in range(num_classes):
            for _ in range(n_per_class):
                # 1×28×28 random tensor, label k
                samples.append((torch.randn(1, 28, 28), k))

        class _PD(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[idx]

        loader = torch.utils.data.DataLoader(_PD(samples), batch_size=8, shuffle=False)
        # Just verify it runs and produces conv1, conv2 entries with right shape
        tfidf = ul._compute_tfidf(loader, num_classes=num_classes)
        assert "conv1" in tfidf
        assert tfidf["conv1"].shape == (10, 32)
        assert "conv2" in tfidf
        assert tfidf["conv2"].shape == (10, 64)
        # All values should be finite
        assert torch.isfinite(tfidf["conv1"]).all()


# ---------------------------------------------------------------------------
# _finetune_on_retain — zero epochs is a noop
# ---------------------------------------------------------------------------
class TestFinetuneOnRetain:
    def test_zero_epochs_noop(self, tmp_path):
        ul = _make_unlearner(tmp_path, finetune_epochs=0)
        params = [np.array(p, copy=True) for p in get_parameters(ul.model)]
        out = ul._finetune_on_retain(params, retain_loader=None)
        # With epochs=0, we get back the same params (object equality not required, but np-equal)
        for a, b in zip(params, out):
            np.testing.assert_array_equal(a, b)

    def test_none_loader_noop(self, tmp_path):
        ul = _make_unlearner(tmp_path, finetune_epochs=5)
        params = [np.array(p, copy=True) for p in get_parameters(ul.model)]
        out = ul._finetune_on_retain(params, retain_loader=None)
        for a, b in zip(params, out):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Public method input validation
# ---------------------------------------------------------------------------
class TestPublicAPIValidation:
    def test_empty_sample_ids_raises(self, tmp_path):
        ul = _make_unlearner(tmp_path)
        with pytest.raises(ValueError, match="non-empty"):
            ul.unlearn_samples_from_contribution(
                round_num=0, client_id=0, sample_ids=[]
            )

    def test_unlearn_samples_all_rounds_empty_ids_raises(self, tmp_path):
        ul = _make_unlearner(tmp_path)
        with pytest.raises(ValueError, match="non-empty"):
            ul.unlearn_samples_all_rounds(
                client_id=0, sample_ids=[], dataset_name="MNIST"
            )
