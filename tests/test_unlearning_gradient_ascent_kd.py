"""Unit tests for GradientAscentKDUnlearning (SCRUB-style)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from contributions_db import ContributionDB
from model import SimpleNet, get_parameters, set_parameters
from unlearning import GradientAscentKDUnlearning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_db(tmp_path):
    return ContributionDB(base_dir=str(tmp_path / "contributions"), dataset_name="MNIST")


def _make_unlearner(tmp_path, **kwargs):
    db = _make_db(tmp_path)
    model = SimpleNet(num_classes=10, num_channels=1, img_size=28)
    return GradientAscentKDUnlearning(
        contribution_db=db,
        model=model,
        dataset_name="MNIST",
        num_clients=2,
        device=torch.device("cpu"),
        batch_size=4,
        **kwargs,
    )


def _tiny_loader(n_per_class=2, num_classes=10, batch_size=4):
    """Tiny labelled loader of (img, label) pairs (no sample IDs, exercises 2-tuple branch)."""
    torch.manual_seed(0)
    samples = []
    for k in range(num_classes):
        for _ in range(n_per_class):
            samples.append((torch.randn(1, 28, 28), k))

    class _DS(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    return torch.utils.data.DataLoader(_DS(samples), batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# _kd_loss
# ---------------------------------------------------------------------------
class TestKDLoss:
    def test_zero_when_student_equals_teacher(self, tmp_path):
        ul = _make_unlearner(tmp_path)
        torch.manual_seed(0)
        logits = torch.randn(4, 10)
        kd = ul._kd_loss(logits, logits)
        assert float(kd.item()) == pytest.approx(0.0, abs=1e-5)

    def test_positive_when_different(self, tmp_path):
        ul = _make_unlearner(tmp_path)
        torch.manual_seed(0)
        a = torch.randn(4, 10)
        b = torch.randn(4, 10)
        kd = ul._kd_loss(a, b)
        assert float(kd.item()) > 0


# ---------------------------------------------------------------------------
# _build_teacher / _build_student
# ---------------------------------------------------------------------------
class TestBuildModels:
    def test_teacher_frozen(self, tmp_path):
        ul = _make_unlearner(tmp_path)
        params = get_parameters(ul.model)
        teacher = ul._build_teacher(params)
        for p in teacher.parameters():
            assert p.requires_grad is False

    def test_student_trainable(self, tmp_path):
        ul = _make_unlearner(tmp_path)
        params = get_parameters(ul.model)
        student = ul._build_student(params)
        for p in student.parameters():
            assert p.requires_grad is True

    def test_teacher_independent_from_self_model(self, tmp_path):
        """Teacher should be a deepcopy — modifying self.model shouldn't change teacher."""
        ul = _make_unlearner(tmp_path)
        params = get_parameters(ul.model)
        teacher = ul._build_teacher(params)
        teacher_first_param = next(teacher.parameters()).clone().detach()
        # Mutate self.model
        with torch.no_grad():
            for p in ul.model.parameters():
                p.zero_()
                break  # mutate just first param
        teacher_after = next(teacher.parameters())
        torch.testing.assert_close(teacher_after, teacher_first_param)


# ---------------------------------------------------------------------------
# _run_finetune
# ---------------------------------------------------------------------------
class TestRunFinetune:
    def test_zero_epochs_noop(self, tmp_path):
        ul = _make_unlearner(tmp_path, epochs=0)
        params = [np.array(p, copy=True) for p in get_parameters(ul.model)]
        forget_loader = _tiny_loader(n_per_class=2)
        retain_loader = _tiny_loader(n_per_class=2)
        new_params, counts = ul._run_finetune(params, forget_loader, retain_loader)
        for a, b in zip(params, new_params):
            np.testing.assert_array_equal(a, b)
        assert counts["forget_steps"] == 0
        assert counts["retain_steps"] == 0

    def test_student_diverges_from_teacher(self, tmp_path):
        """Run a small fine-tune and verify the student's params differ from the input."""
        ul = _make_unlearner(tmp_path, epochs=1, lr=1e-2)
        params = [np.array(p, copy=True) for p in get_parameters(ul.model)]
        forget_loader = _tiny_loader(n_per_class=2)
        retain_loader = _tiny_loader(n_per_class=2)
        new_params, counts = ul._run_finetune(params, forget_loader, retain_loader)
        # At least one tensor should differ
        diffs = [
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(params, new_params)
        ]
        assert max(diffs) > 1e-6, f"Expected student to diverge; max delta = {max(diffs)}"
        assert counts["forget_steps"] > 0
        assert counts["retain_steps"] > 0

    def test_teacher_unchanged_after_finetune(self, tmp_path):
        """The teacher's parameters must not change during the fine-tune loop."""
        ul = _make_unlearner(tmp_path, epochs=1, lr=1e-2)
        params = [np.array(p, copy=True) for p in get_parameters(ul.model)]
        teacher_before = ul._build_teacher(params)
        teacher_snapshot = [p.detach().clone() for p in teacher_before.parameters()]
        # Run finetune (which builds its own teacher internally)
        forget_loader = _tiny_loader(n_per_class=2)
        retain_loader = _tiny_loader(n_per_class=2)
        ul._run_finetune(params, forget_loader, retain_loader)
        # Verify the externally-built teacher is unchanged
        for p, snap in zip(teacher_before.parameters(), teacher_snapshot):
            torch.testing.assert_close(p, snap)

    def test_max_forget_steps_caps_iterations(self, tmp_path):
        """With max_forget_steps=1, forget loop runs exactly 1 step per epoch."""
        ul = _make_unlearner(tmp_path, epochs=2, lr=1e-3, max_forget_steps=1)
        params = [np.array(p, copy=True) for p in get_parameters(ul.model)]
        # Loader with multiple batches
        forget_loader = _tiny_loader(n_per_class=4, batch_size=2)  # → 20 batches
        retain_loader = _tiny_loader(n_per_class=2, batch_size=4)
        _, counts = ul._run_finetune(params, forget_loader, retain_loader)
        # 2 epochs × 1 forget batch = 2 forget steps
        assert counts["forget_steps"] == 2


# ---------------------------------------------------------------------------
# Public API validation
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
