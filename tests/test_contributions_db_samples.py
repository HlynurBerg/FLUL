"""Unit tests for sample-level functionality in ContributionDB."""
import json
import pickle
import pytest
import numpy as np
from pathlib import Path

from contributions_db import (
    ContributionDB,
    _split_metrics_for_manifest,
    contribution_key_from_metadata_path,
    load_contribution_metadata_file,
    parse_round_client_from_contribution_key,
    SAMPLE_MANIFEST_VERSION,
    _SAMPLE_METRIC_KEYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_params():
    return [np.ones((3, 3)), np.zeros(3)]


def _make_db(tmp_path):
    return ContributionDB(base_dir=str(tmp_path / "contributions"), dataset_name="TEST")


# ---------------------------------------------------------------------------
# _split_metrics_for_manifest
# ---------------------------------------------------------------------------

class TestSplitMetricsForManifest:
    def test_empty_metrics_returns_empty_and_none(self):
        lean, manifest = _split_metrics_for_manifest({})
        assert lean == {}
        assert manifest is None

    def test_none_metrics_returns_empty_and_none(self):
        lean, manifest = _split_metrics_for_manifest(None)
        assert lean == {}
        assert manifest is None

    def test_no_sample_keys_returns_all_lean_no_manifest(self):
        metrics = {"loss": 0.5, "accuracy": 0.9}
        lean, manifest = _split_metrics_for_manifest(metrics)
        assert lean == metrics
        assert manifest is None

    def test_sample_ids_goes_to_manifest(self):
        metrics = {"loss": 0.1, "sample_ids": [1, 2, 3]}
        lean, manifest = _split_metrics_for_manifest(metrics)
        assert "sample_ids" not in lean
        assert lean == {"loss": 0.1}
        assert manifest["sample_ids"] == [1, 2, 3]
        assert manifest["sample_manifest_version"] == SAMPLE_MANIFEST_VERSION

    def test_all_sample_keys_go_to_manifest(self):
        sample_data = {k: "value" for k in _SAMPLE_METRIC_KEYS}
        other = {"loss": 0.3}
        metrics = {**sample_data, **other}
        lean, manifest = _split_metrics_for_manifest(metrics)
        for k in _SAMPLE_METRIC_KEYS:
            assert k not in lean
            assert k in manifest
        assert lean == other
        assert manifest["sample_manifest_version"] == SAMPLE_MANIFEST_VERSION

    def test_only_sample_keys_yields_empty_lean(self):
        metrics = {"sample_ids": [0, 1], "training_trace_version": 1}
        lean, manifest = _split_metrics_for_manifest(metrics)
        assert lean == {}
        assert manifest is not None


# ---------------------------------------------------------------------------
# contribution_key_from_metadata_path
# ---------------------------------------------------------------------------

class TestContributionKeyFromMetadataPath:
    def test_strips_metadata_suffix(self):
        path = "/some/dir/round_0001_client_0002_2024-01-01T12-00-00_metadata.json"
        key = contribution_key_from_metadata_path(path)
        assert key == "round_0001_client_0002_2024-01-01T12-00-00"

    def test_plain_stem_when_no_metadata_suffix(self):
        path = "/some/dir/myfile.json"
        key = contribution_key_from_metadata_path(path)
        assert key == "myfile"

    def test_accepts_path_object(self):
        p = Path("/x/round_0003_client_0005_ts_metadata.json")
        key = contribution_key_from_metadata_path(p)
        assert key == "round_0003_client_0005_ts"


# ---------------------------------------------------------------------------
# parse_round_client_from_contribution_key
# ---------------------------------------------------------------------------

class TestParseRoundClientFromContributionKey:
    def test_valid_key(self):
        key = "round_0001_client_0042_2024-01-01T12-00-00"
        rn, cid = parse_round_client_from_contribution_key(key)
        assert rn == 1
        assert cid == 42

    def test_leading_zeros_parsed_correctly(self):
        key = "round_0010_client_0003_ts"
        rn, cid = parse_round_client_from_contribution_key(key)
        assert rn == 10
        assert cid == 3

    def test_missing_round_prefix_returns_none(self):
        rn, cid = parse_round_client_from_contribution_key("client_0001_something")
        assert rn is None and cid is None

    def test_missing_client_segment_returns_none(self):
        rn, cid = parse_round_client_from_contribution_key("round_0001_something_else")
        assert rn is None and cid is None

    def test_non_integer_round_returns_none(self):
        rn, cid = parse_round_client_from_contribution_key("round_abc_client_001_ts")
        assert rn is None and cid is None


# ---------------------------------------------------------------------------
# load_contribution_metadata_file
# ---------------------------------------------------------------------------

class TestLoadContributionMetadataFile:
    def test_loads_plain_metadata(self, tmp_path):
        data = {"round": 1, "client_id": 0, "metrics": {"loss": 0.5}}
        p = tmp_path / "entry_metadata.json"
        p.write_text(json.dumps(data))
        result = load_contribution_metadata_file(p)
        assert result["round"] == 1
        assert result["metrics"]["loss"] == 0.5

    def test_merges_sample_manifest_when_present(self, tmp_path):
        manifest_name = "entry_sample_manifest.json"
        manifest = {"sample_ids": [10, 20], "training_trace_version": 1,
                    "sample_manifest_version": SAMPLE_MANIFEST_VERSION}
        (tmp_path / manifest_name).write_text(json.dumps(manifest))

        data = {
            "round": 1,
            "client_id": 0,
            "metrics": {"loss": 0.5},
            "sample_manifest_file": manifest_name,
        }
        p = tmp_path / "entry_metadata.json"
        p.write_text(json.dumps(data))

        result = load_contribution_metadata_file(p)
        assert result["metrics"]["sample_ids"] == [10, 20]
        assert result["metrics"]["loss"] == 0.5
        assert result["metrics"]["sample_manifest_version"] == SAMPLE_MANIFEST_VERSION

    def test_missing_manifest_file_is_ignored(self, tmp_path):
        data = {
            "round": 1,
            "client_id": 0,
            "metrics": {"loss": 0.3},
            "sample_manifest_file": "nonexistent_sample_manifest.json",
        }
        p = tmp_path / "entry_metadata.json"
        p.write_text(json.dumps(data))
        result = load_contribution_metadata_file(p)
        assert result["metrics"] == {"loss": 0.3}


# ---------------------------------------------------------------------------
# ContributionDB – save_contribution with sample data
# ---------------------------------------------------------------------------

class TestSaveContributionWithSamples:
    def test_sample_manifest_file_created(self, tmp_path):
        db = _make_db(tmp_path)
        metrics = {
            "loss": 0.4,
            "sample_ids": [0, 1, 2],
            "training_trace_version": 1,
        }
        meta_file, _ = db.save_contribution(
            round_num=1, client_id=0, parameters=_dummy_params(),
            num_samples=3, metrics=metrics
        )
        meta = json.loads(meta_file.read_text())
        assert "sample_manifest_file" in meta
        manifest_path = meta_file.parent / meta["sample_manifest_file"]
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["sample_ids"] == [0, 1, 2]
        assert "loss" not in manifest

    def test_lean_metrics_in_metadata_file(self, tmp_path):
        db = _make_db(tmp_path)
        metrics = {"loss": 0.7, "sample_ids": [5, 6]}
        meta_file, _ = db.save_contribution(
            round_num=1, client_id=0, parameters=_dummy_params(),
            num_samples=2, metrics=metrics
        )
        meta = json.loads(meta_file.read_text())
        assert meta["metrics"].get("loss") == 0.7
        assert "sample_ids" not in meta["metrics"]

    def test_no_sample_manifest_when_no_sample_keys(self, tmp_path):
        db = _make_db(tmp_path)
        meta_file, _ = db.save_contribution(
            round_num=1, client_id=0, parameters=_dummy_params(),
            num_samples=10, metrics={"loss": 0.2}
        )
        meta = json.loads(meta_file.read_text())
        assert "sample_manifest_file" not in meta


# ---------------------------------------------------------------------------
# ContributionDB – sample withdrawal
# ---------------------------------------------------------------------------

class TestSampleWithdrawal:
    def _save_one(self, db, round_num=1, client_id=0, sample_ids=None):
        metrics = {"loss": 0.3}
        if sample_ids is not None:
            metrics["sample_ids"] = sample_ids
        return db.save_contribution(
            round_num=round_num, client_id=client_id,
            parameters=_dummy_params(), num_samples=len(sample_ids or [5]),
            metrics=metrics
        )

    def test_withdraw_samples_stores_ids(self, tmp_path):
        db = _make_db(tmp_path)
        meta_file, _ = self._save_one(db, sample_ids=[0, 1, 2, 3])
        meta = json.loads(meta_file.read_text())
        key = meta["contribution_key"]

        result = db.withdraw_samples_from_contribution(key, [1, 3])
        assert sorted(result["sample_ids"]) == [1, 3]

    def test_withdraw_samples_accumulates(self, tmp_path):
        db = _make_db(tmp_path)
        meta_file, _ = self._save_one(db, sample_ids=list(range(10)))
        key = json.loads(meta_file.read_text())["contribution_key"]

        db.withdraw_samples_from_contribution(key, [0, 1])
        db.withdraw_samples_from_contribution(key, [2, 3])
        state = db.get_sample_withdrawal_state(key)
        assert sorted(state["sample_ids"]) == [0, 1, 2, 3]

    def test_withdraw_samples_deduplicates(self, tmp_path):
        db = _make_db(tmp_path)
        meta_file, _ = self._save_one(db, sample_ids=[0, 1, 2])
        key = json.loads(meta_file.read_text())["contribution_key"]

        db.withdraw_samples_from_contribution(key, [1])
        db.withdraw_samples_from_contribution(key, [1, 2])
        state = db.get_sample_withdrawal_state(key)
        assert state["sample_ids"].count(1) == 1

    def test_withdraw_empty_raises(self, tmp_path):
        db = _make_db(tmp_path)
        with pytest.raises(ValueError):
            db.withdraw_samples_from_contribution("any_key", [])

    def test_is_sample_withdrawn_true(self, tmp_path):
        db = _make_db(tmp_path)
        meta_file, _ = self._save_one(db, sample_ids=[10, 20])
        key = json.loads(meta_file.read_text())["contribution_key"]
        db.withdraw_samples_from_contribution(key, [10])
        assert db.is_sample_withdrawn_for_contribution(key, 10) is True
        assert db.is_sample_withdrawn_for_contribution(key, 20) is False

    def test_is_sample_withdrawn_false_when_no_withdrawal(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.is_sample_withdrawn_for_contribution("nonexistent_key", 0) is False

    def test_withdraw_updates_metadata_file(self, tmp_path):
        db = _make_db(tmp_path)
        meta_file, _ = self._save_one(db, sample_ids=[0, 1, 2])
        key = json.loads(meta_file.read_text())["contribution_key"]
        db.withdraw_samples_from_contribution(key, [0, 2])
        meta = json.loads(meta_file.read_text())
        assert sorted(meta["withdrawn_sample_ids"]) == [0, 2]

    def test_withdraw_adds_history_entry(self, tmp_path):
        db = _make_db(tmp_path)
        meta_file, _ = self._save_one(db, sample_ids=[5, 6, 7])
        key = json.loads(meta_file.read_text())["contribution_key"]
        db.withdraw_samples_from_contribution(key, [5], reason="privacy")
        history = db.withdrawals["sample_withdrawals"]["history"]
        assert len(history) == 1
        assert history[0]["event"] == "withdraw_samples"
        assert history[0]["reason"] == "privacy"

    def test_withdrawal_persists_across_db_reload(self, tmp_path):
        db = _make_db(tmp_path)
        meta_file, _ = self._save_one(db, sample_ids=[1, 2, 3])
        key = json.loads(meta_file.read_text())["contribution_key"]
        db.withdraw_samples_from_contribution(key, [2])

        db2 = ContributionDB(
            base_dir=str(tmp_path / "contributions"), dataset_name="TEST"
        )
        assert db2.is_sample_withdrawn_for_contribution(key, 2) is True
        assert db2.is_sample_withdrawn_for_contribution(key, 1) is False


# ---------------------------------------------------------------------------
# ContributionDB – restore_samples_from_contribution
# ---------------------------------------------------------------------------

class TestRestoreSamples:
    def _save_and_withdraw(self, db, sample_ids, withdraw_ids):
        metrics = {"loss": 0.2, "sample_ids": sample_ids}
        meta_file, _ = db.save_contribution(
            round_num=1, client_id=0, parameters=_dummy_params(),
            num_samples=len(sample_ids), metrics=metrics
        )
        key = json.loads(meta_file.read_text())["contribution_key"]
        db.withdraw_samples_from_contribution(key, withdraw_ids)
        return key

    def test_restore_specific_samples(self, tmp_path):
        db = _make_db(tmp_path)
        key = self._save_and_withdraw(db, [0, 1, 2, 3], [0, 1, 2])
        db.restore_samples_from_contribution(key, [1])
        state = db.get_sample_withdrawal_state(key)
        assert 1 not in state["sample_ids"]
        assert 0 in state["sample_ids"]
        assert 2 in state["sample_ids"]

    def test_restore_all_samples(self, tmp_path):
        db = _make_db(tmp_path)
        key = self._save_and_withdraw(db, [0, 1, 2], [0, 1, 2])
        db.restore_samples_from_contribution(key, all_samples=True)
        assert db.get_sample_withdrawal_state(key) is None

    def test_restore_last_sample_clears_active_entry(self, tmp_path):
        db = _make_db(tmp_path)
        key = self._save_and_withdraw(db, [7], [7])
        db.restore_samples_from_contribution(key, [7])
        assert db.get_sample_withdrawal_state(key) is None

    def test_restore_nonexistent_returns_false(self, tmp_path):
        db = _make_db(tmp_path)
        result = db.restore_samples_from_contribution("no_such_key", all_samples=True)
        assert result is False

    def test_restore_without_ids_or_all_raises(self, tmp_path):
        db = _make_db(tmp_path)
        key = self._save_and_withdraw(db, [0, 1], [0])
        with pytest.raises(ValueError):
            db.restore_samples_from_contribution(key)

    def test_restore_adds_history_entry(self, tmp_path):
        db = _make_db(tmp_path)
        key = self._save_and_withdraw(db, [0, 1], [0, 1])
        db.restore_samples_from_contribution(key, [0])
        history = db.withdrawals["sample_withdrawals"]["history"]
        restore_events = [e for e in history if e["event"] == "restore_samples"]
        assert len(restore_events) == 1

    def test_restore_updates_metadata_file(self, tmp_path):
        db = _make_db(tmp_path)
        key = self._save_and_withdraw(db, [0, 1, 2], [0, 1, 2])
        round_dir = db.contributions_dir / "round_0001" / "client_0000"
        meta_file = round_dir / f"{key}_metadata.json"
        db.restore_samples_from_contribution(key, [0])
        meta = json.loads(meta_file.read_text())
        assert 0 not in meta["withdrawn_sample_ids"]
        assert 1 in meta["withdrawn_sample_ids"]


# ---------------------------------------------------------------------------
# ContributionDB – enrich_contribution_view
# ---------------------------------------------------------------------------

class TestEnrichContributionView:
    def test_adds_withdrawn_sample_ids_from_active(self, tmp_path):
        db = _make_db(tmp_path)
        meta_file, _ = db.save_contribution(
            round_num=1, client_id=0, parameters=_dummy_params(),
            num_samples=5, metrics={"sample_ids": [0, 1, 2, 3, 4]}
        )
        key = json.loads(meta_file.read_text())["contribution_key"]
        db.withdraw_samples_from_contribution(key, [2, 4])

        contrib = {"contribution_key": key}
        db.enrich_contribution_view(contrib)
        assert contrib["withdrawn_sample_ids"] == [2, 4]

    def test_defaults_to_empty_list_when_no_withdrawal(self, tmp_path):
        db = _make_db(tmp_path)
        contrib = {"contribution_key": "round_0001_client_0000_ts"}
        db.enrich_contribution_view(contrib)
        assert contrib["withdrawn_sample_ids"] == []

    def test_sets_contribution_key_from_path(self, tmp_path):
        db = _make_db(tmp_path)
        path = tmp_path / "round_0001_client_0000_ts_metadata.json"
        path.write_text("{}")
        contrib = {}
        db.enrich_contribution_view(contrib, metadata_path=path)
        assert contrib["contribution_key"] == "round_0001_client_0000_ts"


# ---------------------------------------------------------------------------
# ContributionDB – get_history sample withdrawal visibility
# ---------------------------------------------------------------------------

class TestGetHistorySampleWithdrawals:
    def test_history_includes_withdrawn_sample_ids(self, tmp_path):
        db = _make_db(tmp_path)
        meta_file, _ = db.save_contribution(
            round_num=1, client_id=0, parameters=_dummy_params(),
            num_samples=3, metrics={"sample_ids": [10, 20, 30]}
        )
        key = json.loads(meta_file.read_text())["contribution_key"]
        db.withdraw_samples_from_contribution(key, [10, 30])

        history = db.get_history(client_id=0, round_num=1)
        assert len(history) == 1
        assert sorted(history[0]["withdrawn_sample_ids"]) == [10, 30]

    def test_history_empty_withdrawn_sample_ids_when_none_withdrawn(self, tmp_path):
        db = _make_db(tmp_path)
        db.save_contribution(
            round_num=1, client_id=0, parameters=_dummy_params(),
            num_samples=2, metrics={}
        )
        history = db.get_history(client_id=0)
        assert history[0]["withdrawn_sample_ids"] == []
