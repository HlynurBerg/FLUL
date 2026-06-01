"""Contribution database for tracking client contributions in federated learning."""
import os
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

# Large / per-sample fields stored in a sibling *_sample_manifest.json (v1).
SAMPLE_MANIFEST_VERSION = 1
_SAMPLE_METRIC_KEYS = frozenset(
    {
        "sample_ids",
        "training_trace",
        "training_trace_version",
        "sample_id_scheme_version",
    }
)


def _split_metrics_for_manifest(metrics: dict):
    """Return (lean_metrics, manifest_dict_or_None).

    The client serializes ``training_trace`` to a JSON string before sending
    (Flower 1.22 rejects nested dicts in metrics). Parse it back here so
    downstream consumers see the legacy nested-dict shape.
    """
    if not metrics:
        return {}, None
    manifest = {k: metrics[k] for k in _SAMPLE_METRIC_KEYS if k in metrics}
    lean = {k: v for k, v in metrics.items() if k not in _SAMPLE_METRIC_KEYS}
    if not manifest:
        return lean, None
    manifest = dict(manifest)
    for json_str_key in ("training_trace", "sample_ids"):
        if isinstance(manifest.get(json_str_key), str):
            try:
                manifest[json_str_key] = json.loads(manifest[json_str_key])
            except (TypeError, json.JSONDecodeError):
                pass  # leave as-is; downstream loaders can detect malformed values
    manifest["sample_manifest_version"] = SAMPLE_MANIFEST_VERSION
    return lean, manifest


def load_contribution_metadata_file(metadata_path) -> dict:
    """
    Load ``*_metadata.json`` and merge optional ``*_sample_manifest.json`` when
    ``sample_manifest_file`` is set (same directory, filename from metadata).

    Returns a single dict matching the legacy shape (full ``metrics``), so callers
    need not know whether data was split across files.
    """
    metadata_path = Path(metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    client_dir = metadata_path.parent
    manifest_name = data.get("sample_manifest_file")
    if manifest_name:
        manifest_path = client_dir / manifest_name
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            merged = dict(data.get("metrics") or {})
            merged.update(manifest)
            data["metrics"] = merged
    return data


def contribution_key_from_metadata_path(metadata_path) -> str:
    """Return the contribution key (base filename stem) from ``*_metadata.json`` path."""
    name = Path(metadata_path).name
    if name.endswith("_metadata.json"):
        return name[: -len("_metadata.json")]
    return Path(metadata_path).stem


def parse_round_client_from_contribution_key(contribution_key: str):
    """
    Parse ``round`` and ``client_id`` from a contribution key
    ``round_RRRR_client_CCCC_<timestamp>``.
    Returns (round_num, client_id) or (None, None) if not parseable.
    """
    if not contribution_key.startswith("round_") or "_client_" not in contribution_key:
        return None, None
    try:
        r_end = contribution_key.index("_client_")
        round_num = int(contribution_key[6:r_end])
        tail = contribution_key[r_end + len("_client_") :]
        c_end = tail.index("_")
        client_id = int(tail[:c_end])
        return round_num, client_id
    except (ValueError, IndexError):
        return None, None


class ContributionDB:
    """Database for storing client contributions during federated learning."""
    
    def __init__(self, base_dir="contributions", dataset_name="MNIST"):
        """
        Initialize the contribution database.
        
        Args:
            base_dir: Base directory for storing contributions
            dataset_name: Name of the dataset being used
        """
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name
        self.contributions_dir = self.base_dir / dataset_name
        self.contributions_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.metadata_dir = self.contributions_dir / "metadata"
        self.parameters_dir = self.contributions_dir / "parameters"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.parameters_dir.mkdir(parents=True, exist_ok=True)
        
        # Track statistics
        self.stats_file = self.contributions_dir / "statistics.json"
        self._load_statistics()
        
        # Track withdrawals (like git history)
        self.withdrawals_file = self.contributions_dir / "withdrawals.json"
        self._load_withdrawals()
    
    def _load_statistics(self):
        """Load existing statistics or create new ones."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    loaded_stats = json.load(f)
                    # Convert lists back to sets for internal use
                    self.stats = {
                        "total_contributions": loaded_stats.get("total_contributions", 0),
                        "total_rounds": loaded_stats.get("total_rounds", 0),
                        "clients_participated": set(loaded_stats.get("clients_participated", [])),
                        "rounds": {}
                    }
                    # Convert round data back to sets
                    for round_num_str, round_data in loaded_stats.get("rounds", {}).items():
                        self.stats["rounds"][int(round_num_str)] = {
                            "contributions": round_data.get("contributions", 0),
                            "clients": set(round_data.get("clients", []))
                        }
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"[Contribution DB] WARNING: Failed to load statistics.json: {e}")
                print(f"[Contribution DB] Attempting to recover statistics from contribution files...")
                
                # Try to recover statistics by scanning contribution directories
                recovered_stats = {
                    "total_contributions": 0,
                    "total_rounds": 0,
                    "clients_participated": set(),
                    "rounds": {}
                }
                
                # Scan round directories
                round_dirs = [d for d in self.contributions_dir.iterdir() 
                            if d.is_dir() and d.name.startswith('round_')]
                
                for round_dir in round_dirs:
                    try:
                        round_num = int(round_dir.name.split('_')[1])
                        recovered_stats["total_rounds"] = max(recovered_stats["total_rounds"], round_num)
                        
                        # Count contributions in this round
                        round_contributions = 0
                        round_clients = set()
                        
                        for client_dir in round_dir.iterdir():
                            if client_dir.is_dir() and client_dir.name.startswith('client_'):
                                try:
                                    client_id = int(client_dir.name.split('_')[1])
                                    metadata_files = list(client_dir.glob("*_metadata.json"))
                                    round_contributions += len(metadata_files)
                                    if metadata_files:
                                        round_clients.add(client_id)
                                        recovered_stats["clients_participated"].add(client_id)
                                except (ValueError, IndexError):
                                    continue
                        
                        if round_contributions > 0:
                            recovered_stats["rounds"][round_num] = {
                                "contributions": round_contributions,
                                "clients": round_clients
                            }
                            recovered_stats["total_contributions"] += round_contributions
                    except (ValueError, IndexError):
                        continue
                
                # Backup the corrupted file
                if self.stats_file.exists():
                    backup_file = self.stats_file.with_suffix('.json.bak')
                    import shutil
                    shutil.copy2(self.stats_file, backup_file)
                    print(f"[Contribution DB] Backed up corrupted file to: {backup_file}")
                
                # Use recovered stats or initialize empty
                if recovered_stats["total_contributions"] > 0:
                    print(f"[Contribution DB] Recovered {recovered_stats['total_contributions']} contributions from {recovered_stats['total_rounds']} rounds")
                    self.stats = recovered_stats
                else:
                    print(f"[Contribution DB] No contributions found, creating new statistics file")
                    self.stats = {
                        "total_contributions": 0,
                        "total_rounds": 0,
                        "clients_participated": set(),
                        "rounds": {}
                    }
                
                # Save the recovered/new stats
                self._save_statistics()
        else:
            self.stats = {
                "total_contributions": 0,
                "total_rounds": 0,
                "clients_participated": set(),
                "rounds": {}
            }
    
    def _load_withdrawals(self):
        """Load withdrawal history."""
        if self.withdrawals_file.exists():
            with open(self.withdrawals_file, 'r') as f:
                self.withdrawals = json.load(f)
        else:
            self.withdrawals = {
                "withdrawn_clients": {},  # client_id -> {"withdrawn_at": timestamp, "reason": optional}
                "withdrawal_history": []  # List of withdrawal events
            }
        self._ensure_sample_withdrawals_schema()
    
    def _ensure_sample_withdrawals_schema(self):
        """Ensure withdrawals.json has sample-level tracking (parallel to full-client withdrawal)."""
        sw = self.withdrawals.setdefault("sample_withdrawals", {})
        sw.setdefault("active", {})  # contribution_key -> { sample_ids, withdrawn_at, reason }
        sw.setdefault("history", [])  # list of events
    
    def _save_statistics(self):
        """Save statistics to file."""
        # Convert sets to lists for JSON serialization
        stats_to_save = {
            "total_contributions": self.stats["total_contributions"],
            "total_rounds": self.stats["total_rounds"],
            "clients_participated": list(self.stats["clients_participated"]) if isinstance(self.stats["clients_participated"], set) else self.stats["clients_participated"],
            "rounds": {}
        }
        
        # Convert sets in rounds to lists
        for round_num, round_data in self.stats["rounds"].items():
            round_info = {
                "contributions": round_data.get("contributions", 0),
                "clients": list(round_data["clients"]) if isinstance(round_data.get("clients"), set) else round_data.get("clients", [])
            }
            stats_to_save["rounds"][str(round_num)] = round_info
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
    
    def _save_withdrawals(self):
        """Save withdrawal history to file."""
        with open(self.withdrawals_file, 'w') as f:
            json.dump(self.withdrawals, f, indent=2)
    
    def save_contribution(self, round_num, client_id, parameters, num_samples, metrics=None):
        """
        Save a client contribution to the database.
        
        Args:
            round_num: Current federated learning round
            client_id: ID of the contributing client
            parameters: Model parameters (list of numpy arrays)
            num_samples: Number of training samples used by client
            metrics: Optional dictionary of training metrics (loss, accuracy, etc.)
        """
        timestamp = datetime.now().isoformat()
        
        # Create round directory
        round_dir = self.contributions_dir / f"round_{round_num:04d}"
        round_dir.mkdir(exist_ok=True)
        
        # Create client directory within round
        client_dir = round_dir / f"client_{client_id:04d}"
        client_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp_short = timestamp.replace(':', '-').split('.')[0]
        base_filename = f"round_{round_num:04d}_client_{client_id:04d}_{timestamp_short}"
        
        # Check if client is withdrawn
        is_withdrawn = self.is_client_withdrawn(client_id)
        
        lean_metrics, sample_manifest = _split_metrics_for_manifest(metrics or {})
        
        # Save metadata (lean aggregate metrics; per-sample data in *_sample_manifest.json)
        metadata = {
            "round": round_num,
            "client_id": client_id,
            "contribution_key": base_filename,
            "timestamp": timestamp,
            "num_samples": num_samples,
            "parameters_count": len(parameters),
            "metrics": lean_metrics,
            "parameters_file": f"{base_filename}_params.pkl",
            "withdrawn": is_withdrawn,
        }
        if sample_manifest is not None:
            manifest_filename = f"{base_filename}_sample_manifest.json"
            metadata["sample_manifest_file"] = manifest_filename
            manifest_path = client_dir / manifest_filename
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(sample_manifest, f, indent=2)
        
        active_sw = self.withdrawals.get("sample_withdrawals", {}).get("active", {}).get(base_filename)
        if active_sw:
            metadata["withdrawn_sample_ids"] = sorted(int(x) for x in active_sw.get("sample_ids", []))
        
        metadata_file = client_dir / f"{base_filename}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Save parameters
        params_file = client_dir / f"{base_filename}_params.pkl"
        with open(params_file, 'wb') as f:
            pickle.dump(parameters, f)
        
        # Update statistics
        self.stats["total_contributions"] += 1
        self.stats["clients_participated"].add(client_id)
        
        if round_num not in self.stats["rounds"]:
            self.stats["rounds"][round_num] = {
                "contributions": 0,
                "clients": set()
            }
        
        self.stats["rounds"][round_num]["contributions"] += 1
        self.stats["rounds"][round_num]["clients"].add(client_id)
        self.stats["total_rounds"] = max(self.stats["total_rounds"], round_num)
        
        self._save_statistics()
        
        print(f"[Contribution DB] Saved contribution: Round {round_num}, Client {client_id}, {num_samples} samples")
        
        return metadata_file, params_file
    
    def save_aggregated_model(self, round_num, parameters, metrics=None, suffix=None):
        """
        Save the aggregated (global) model after each round.
        
        Args:
            round_num: Current federated learning round
            parameters: Aggregated model parameters
            metrics: Optional dictionary of evaluation metrics
            suffix: Optional suffix to add to filename (e.g., "unlearned_gradient_original")
        """
        timestamp = datetime.now().isoformat()
        round_dir = self.contributions_dir / f"round_{round_num:04d}"
        round_dir.mkdir(exist_ok=True)
        
        timestamp_short = timestamp.replace(':', '-').split('.')[0]
        base_filename = f"round_{round_num:04d}_aggregated_{timestamp_short}"
        
        # Add suffix if provided
        if suffix:
            base_filename = f"{base_filename}_{suffix}"
        
        # Save metadata
        metadata = {
            "round": round_num,
            "timestamp": timestamp,
            "type": "aggregated",
            "parameters_count": len(parameters),
            "metrics": metrics or {},
            "parameters_file": f"{base_filename}_params.pkl",
            "suffix": suffix
        }
        
        metadata_file = round_dir / f"{base_filename}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save parameters
        params_file = round_dir / f"{base_filename}_params.pkl"
        with open(params_file, 'wb') as f:
            pickle.dump(parameters, f)
        
        model_type = f" ({suffix})" if suffix else ""
        print(f"[Contribution DB] Saved aggregated model: Round {round_num}{model_type}")
        
        return metadata_file, params_file
    
    def save_initial_global_parameters(self, parameters):
        """
        Save the server global model **before** round 0 (incoming weights for all clients
        in the first federated round). Required for replay-based per-sample gradient recovery.
        """
        path = self.contributions_dir / "initial_global_params.pkl"
        meta = {
            "type": "initial_global",
            "timestamp": datetime.now().isoformat(),
            "parameters_count": len(parameters),
        }
        with open(self.contributions_dir / "initial_global_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        with open(path, "wb") as f:
            pickle.dump(parameters, f)
        print(f"[Contribution DB] Saved initial global parameters: {path}")
        return path
    
    def load_initial_global_parameters(self):
        """Load list of numpy arrays for the initial global model, or None if missing."""
        path = self.contributions_dir / "initial_global_params.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def get_latest_aggregated_parameters(self, round_num: int):
        """
        Load the most recently written aggregated global model for ``round_num``
        (the global weights **after** that federated round).
        """
        round_dir = self.contributions_dir / f"round_{round_num:04d}"
        if not round_dir.is_dir():
            return None
        agg_files = list(round_dir.glob("*_aggregated_*_params.pkl"))
        if not agg_files:
            return None
        latest = max(agg_files, key=lambda p: p.stat().st_mtime)
        with open(latest, "rb") as f:
            return pickle.load(f)
    
    def load_incoming_global_for_round(self, round_num: int):
        """
        Global model weights **at the start** of federated round ``round_num``
        (what clients receive before local training).

        - Round 0: initial global model (``save_initial_global_parameters``).
        - Round R>0: aggregated model after round R-1.
        """
        if round_num == 0:
            return self.load_initial_global_parameters()
        return self.get_latest_aggregated_parameters(round_num - 1)
    
    def get_contribution_info(self, round_num, client_id):
        """
        Retrieve information about a specific contribution.

        If ``sample_manifest_file`` is present, the sibling ``*_sample_manifest.json``
        is loaded and its fields are merged into ``metrics`` (same shape as legacy
        single-file metadata).
        
        Args:
            round_num: Round number
            client_id: Client ID
            
        Returns:
            Dictionary with contribution metadata or None if not found
        """
        round_dir = self.contributions_dir / f"round_{round_num:04d}"
        client_dir = round_dir / f"client_{client_id:04d}"
        
        if not client_dir.exists():
            return None
        
        # Find the most recent metadata file
        metadata_files = list(client_dir.glob("*_metadata.json"))
        if not metadata_files:
            return None
        
        latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)
        data = load_contribution_metadata_file(latest_metadata)
        self.enrich_contribution_view(data, latest_metadata)
        return data
    
    def enrich_contribution_view(self, contrib: dict, metadata_path=None):
        """
        Attach ``contribution_key`` (if missing) and ``withdrawn_sample_ids`` from
        ``sample_withdrawals.active`` in withdrawals.json (source of truth).
        """
        if metadata_path is not None:
            if not contrib.get("contribution_key"):
                contrib["contribution_key"] = contribution_key_from_metadata_path(metadata_path)
        key = contrib.get("contribution_key")
        if not key:
            return
        self._ensure_sample_withdrawals_schema()
        entry = self.withdrawals["sample_withdrawals"]["active"].get(key)
        if entry:
            contrib["withdrawn_sample_ids"] = sorted(int(x) for x in entry.get("sample_ids", []))
        else:
            contrib.setdefault("withdrawn_sample_ids", [])
    
    def resolve_contribution_key(self, round_num, client_id, contribution_key=None):
        """Resolve contribution key; default is latest contribution for (round, client)."""
        if contribution_key:
            return contribution_key
        c = self.get_contribution_info(round_num, client_id)
        if c and c.get("contribution_key"):
            return c["contribution_key"]
        raise ValueError(f"No contribution found for round {round_num} client {client_id}")
    
    def withdraw_samples_from_contribution(self, contribution_key, sample_ids, reason=None):
        """
        Mark specific global sample IDs as withdrawn for one contribution (atomic unit is still
        one saved contribution; this withdraws a **part** of it). Independent of full-client
        ``withdrawn`` flag.
        """
        self._ensure_sample_withdrawals_schema()
        sample_ids = sorted({int(x) for x in sample_ids})
        if not sample_ids:
            raise ValueError("sample_ids must be non-empty")
        sw = self.withdrawals["sample_withdrawals"]
        prev = sw["active"].get(contribution_key, {})
        prev_ids = set(prev.get("sample_ids", []))
        prev_ids |= set(sample_ids)
        merged = sorted(prev_ids)
        ts = datetime.now().isoformat()
        sw["active"][contribution_key] = {
            "sample_ids": merged,
            "withdrawn_at": ts,
            "reason": reason,
        }
        sw["history"].append(
            {
                "event": "withdraw_samples",
                "contribution_key": contribution_key,
                "sample_ids": list(sample_ids),
                "withdrawn_at": ts,
                "reason": reason,
            }
        )
        self._save_withdrawals()
        self._set_denormalized_withdrawn_sample_ids(contribution_key, merged)
        print(
            f"[Contribution DB] Sample withdrawal on {contribution_key}: "
            f"{len(merged)} sample id(s) withdrawn in total"
        )
        return sw["active"][contribution_key]
    
    def restore_samples_from_contribution(self, contribution_key, sample_ids=None, all_samples=False):
        """
        Remove sample IDs from the active sample-withdrawal set for this contribution.
        If ``all_samples`` is True, clear all sample withdrawals for the contribution.
        """
        self._ensure_sample_withdrawals_schema()
        sw = self.withdrawals["sample_withdrawals"]
        if contribution_key not in sw["active"]:
            print(f"[Contribution DB] No active sample withdrawal for {contribution_key}")
            return False
        if all_samples:
            del sw["active"][contribution_key]
            remaining = []
        else:
            if not sample_ids:
                raise ValueError("Pass sample_ids or use all_samples=True")
            to_remove = {int(x) for x in sample_ids}
            cur = set(sw["active"][contribution_key].get("sample_ids", []))
            cur -= to_remove
            if not cur:
                del sw["active"][contribution_key]
                remaining = []
            else:
                sw["active"][contribution_key]["sample_ids"] = sorted(cur)
                remaining = sorted(cur)
        ts = datetime.now().isoformat()
        sw["history"].append(
            {
                "event": "restore_samples",
                "contribution_key": contribution_key,
                "sample_ids": None if all_samples else list(sample_ids),
                "all_samples": all_samples,
                "restored_at": ts,
            }
        )
        self._save_withdrawals()
        self._set_denormalized_withdrawn_sample_ids(contribution_key, remaining)
        return True
    
    def get_sample_withdrawal_state(self, contribution_key: str):
        """Return the active sample-withdrawal record for a contribution, or None."""
        self._ensure_sample_withdrawals_schema()
        return self.withdrawals["sample_withdrawals"]["active"].get(contribution_key)
    
    def is_sample_withdrawn_for_contribution(self, contribution_key: str, sample_id: int) -> bool:
        st = self.get_sample_withdrawal_state(contribution_key)
        if not st:
            return False
        return int(sample_id) in set(st.get("sample_ids", []))
    
    def _set_denormalized_withdrawn_sample_ids(self, contribution_key: str, sample_ids: list):
        """Mirror withdrawn_sample_ids on *_metadata.json for grep-friendly inspection."""
        rn, cid = parse_round_client_from_contribution_key(contribution_key)
        if rn is None:
            return
        path = (
            self.contributions_dir
            / f"round_{rn:04d}"
            / f"client_{cid:04d}"
            / f"{contribution_key}_metadata.json"
        )
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["withdrawn_sample_ids"] = list(sample_ids)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    
    def get_statistics(self):
        """Get overall statistics about contributions."""
        stats = self.stats.copy()
        stats["clients_participated"] = list(stats["clients_participated"])
        # Convert sets in rounds
        for round_num in stats["rounds"]:
            stats["rounds"][round_num]["clients"] = list(stats["rounds"][round_num]["clients"])
        return stats

    def find_rounds_containing_samples(self, client_id: int, sample_ids: list) -> dict:
        """
        Find all rounds where a client's contribution contained specific sample IDs.

        Scans the sample manifest (``*_sample_manifest.json``) for each contribution
        by ``client_id`` and returns the subset of ``sample_ids`` that appears in each
        round.  If a contribution has no sample manifest (training trace was disabled),
        the full ``sample_ids`` list is assumed present for that round so that unlearning
        is conservative rather than silent.

        Args:
            client_id: The client whose contributions to scan.
            sample_ids: Global sample IDs to search for.

        Returns:
            Dict mapping ``round_num`` -> list of sample IDs found in that round.
            Only rounds where at least one target sample appears (or where no manifest
            exists) are included.
        """
        target = set(int(s) for s in sample_ids)
        result = {}

        all_contributions = self.list_contributions(exclude_withdrawn=False)
        client_contributions = [c for c in all_contributions if c["client_id"] == client_id]

        for contrib in sorted(client_contributions, key=lambda c: c["round"]):
            round_num = contrib["round"]
            metrics = contrib.get("metrics") or {}
            manifest_sample_ids = metrics.get("sample_ids")

            if manifest_sample_ids is None:
                # No sample manifest — conservative: assume all target samples present
                result[round_num] = sorted(target)
            else:
                found = sorted(target & set(int(s) for s in manifest_sample_ids))
                if found:
                    result[round_num] = found

        return result

    def withdraw_client(self, client_id, reason=None):
        """
        Mark a client as withdrawn (like git, contributions are preserved but marked).
        
        Args:
            client_id: ID of the client to withdraw
            reason: Optional reason for withdrawal
        """
        timestamp = datetime.now().isoformat()
        
        withdrawal_event = {
            "client_id": client_id,
            "withdrawn_at": timestamp,
            "reason": reason,
            "affects_rounds": []
        }
        
        # Find all rounds where this client contributed
        all_contributions = self.list_contributions()
        affected_rounds = set()
        for contrib in all_contributions:
            if contrib["client_id"] == client_id:
                affected_rounds.add(contrib["round"])
        
        withdrawal_event["affects_rounds"] = sorted(list(affected_rounds))
        
        # Record withdrawal
        self.withdrawals["withdrawn_clients"][str(client_id)] = {
            "withdrawn_at": timestamp,
            "reason": reason,
            "affects_rounds": withdrawal_event["affects_rounds"]
        }
        
        self.withdrawals["withdrawal_history"].append(withdrawal_event)
        self._save_withdrawals()
        
        # Update metadata files to mark contributions as withdrawn
        for rnd in affected_rounds:
            round_dir = self.contributions_dir / f"round_{rnd:04d}"
            client_dir = round_dir / f"client_{client_id:04d}"
            if client_dir.exists():
                for metadata_file in client_dir.glob("*_metadata.json"):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    metadata["withdrawn"] = True
                    metadata["withdrawn_at"] = timestamp
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
        
        print(f"[Contribution DB] Client {client_id} marked as withdrawn (affects {len(affected_rounds)} rounds)")
        return withdrawal_event
    
    def restore_client(self, client_id):
        """
        Restore a withdrawn client (undo withdrawal).
        
        Args:
            client_id: ID of the client to restore
        """
        if str(client_id) not in self.withdrawals["withdrawn_clients"]:
            print(f"[Contribution DB] Client {client_id} is not withdrawn")
            return False
        
        # Get withdrawal info before deleting
        withdrawal_info = self.withdrawals["withdrawn_clients"][str(client_id)]
        
        # Remove from withdrawn clients
        del self.withdrawals["withdrawn_clients"][str(client_id)]
        self._save_withdrawals()
        
        # Update metadata files
        if withdrawal_info:
            for rnd in withdrawal_info.get("affects_rounds", []):
                round_dir = self.contributions_dir / f"round_{rnd:04d}"
                client_dir = round_dir / f"client_{client_id:04d}"
                if client_dir.exists():
                    for metadata_file in client_dir.glob("*_metadata.json"):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        metadata["withdrawn"] = False
                        if "withdrawn_at" in metadata:
                            del metadata["withdrawn_at"]
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
        
        print(f"[Contribution DB] Client {client_id} restored")
        return True
    
    def is_client_withdrawn(self, client_id):
        """Check if a client is currently withdrawn."""
        return str(client_id) in self.withdrawals["withdrawn_clients"]
    
    def get_withdrawn_clients(self):
        """Get list of all withdrawn clients."""
        return list(self.withdrawals["withdrawn_clients"].keys())
    
    def get_withdrawal_history(self):
        """Get complete withdrawal history."""
        return self.withdrawals["withdrawal_history"]
    
    def list_contributions(self, round_num=None, exclude_withdrawn=False):
        """
        List all contributions, optionally filtered by round and excluding withdrawn clients.
        
        Args:
            round_num: Optional round number to filter by
            exclude_withdrawn: If True, exclude contributions from withdrawn clients
            
        Returns:
            List of contribution metadata dictionaries
        """
        contributions = []
        
        if round_num is not None:
            rounds_to_check = [round_num]
        else:
            # Parse round numbers from directory names, handling errors gracefully
            rounds_to_check = []
            for d in self.contributions_dir.iterdir():
                if d.is_dir() and d.name.startswith('round_'):
                    try:
                        # Extract round number from "round_XXXX" format
                        round_num_val = int(d.name.split('_')[1])
                        rounds_to_check.append(round_num_val)
                    except (ValueError, IndexError):
                        # Skip directories that don't match the expected format
                        continue
            rounds_to_check = sorted(rounds_to_check)
        
        for rnd in rounds_to_check:
            round_dir = self.contributions_dir / f"round_{rnd:04d}"
            if not round_dir.exists():
                continue
            
            # Get client contributions
            for client_dir in round_dir.iterdir():
                if client_dir.is_dir() and client_dir.name.startswith('client_'):
                    metadata_files = list(client_dir.glob("*_metadata.json"))
                    for metadata_file in metadata_files:
                        contrib = load_contribution_metadata_file(metadata_file)
                        self.enrich_contribution_view(contrib, metadata_file)
                        
                        # Check withdrawal status
                        client_id = contrib["client_id"]
                        if exclude_withdrawn and self.is_client_withdrawn(client_id):
                            continue
                        
                        # Ensure withdrawal status is up to date
                        contrib["withdrawn"] = self.is_client_withdrawn(client_id)
                        contributions.append(contrib)
        
        return contributions
    
    def recalculate_aggregated_model(self, round_num, exclude_withdrawn=True):
        """
        Recalculate aggregated model for a round excluding withdrawn clients.
        This implements the "git-like" behavior of rebuilding without withdrawn contributions.

        When a contribution has withdrawn samples (``withdrawn_sample_ids``), its
        fed-averaging weight is reduced to ``num_samples - len(withdrawn_sample_ids)``
        so the clean recomputation path is consistent with per-sample withdrawals.
        
        Args:
            round_num: Round number to recalculate
            exclude_withdrawn: If True, exclude withdrawn clients from aggregation
            
        Returns:
            Recalculated aggregated parameters or None if insufficient contributions
        """
        # Get all contributions for this round
        contributions = self.list_contributions(round_num=round_num, exclude_withdrawn=exclude_withdrawn)
        
        if not contributions:
            print(f"[Contribution DB] No contributions found for round {round_num}")
            return None
        
        # Load parameters and calculate weighted average
        total_samples = 0
        weighted_params = None
        
        for contrib in contributions:
            client_id = contrib["client_id"]
            
            # Load parameters
            round_dir = self.contributions_dir / f"round_{round_num:04d}"
            client_dir = round_dir / f"client_{client_id:04d}"
            params_file = client_dir / contrib["parameters_file"]
            
            if not params_file.exists():
                continue
            
            with open(params_file, 'rb') as f:
                params = pickle.load(f)
            
            withdrawn_n = len(contrib.get("withdrawn_sample_ids") or [])
            num_samples = max(1, contrib["num_samples"] - withdrawn_n)

            # Initialize or accumulate weighted parameters
            if weighted_params is None:
                weighted_params = [p * num_samples for p in params]
            else:
                for i in range(len(weighted_params)):
                    weighted_params[i] += params[i] * num_samples

            total_samples += num_samples
        
        if total_samples == 0 or weighted_params is None:
            print(f"[Contribution DB] Insufficient contributions for round {round_num}")
            return None
        
        # Normalize by total samples (Federated Averaging)
        aggregated_params = [p / total_samples for p in weighted_params]
        
        print(f"[Contribution DB] Recalculated aggregated model for round {round_num} "
              f"(excluded {len([c for c in self.list_contributions(round_num=round_num) if self.is_client_withdrawn(c['client_id'])])} withdrawn clients)")
        
        return aggregated_params
    
    def get_history(self, client_id=None, round_num=None):
        """
        Get git-like history of contributions.
        
        Args:
            client_id: Optional client ID to filter by
            round_num: Optional round number to filter by
            
        Returns:
            List of contribution history entries with withdrawal status
        """
        contributions = self.list_contributions(round_num=round_num, exclude_withdrawn=False)
        
        if client_id is not None:
            contributions = [c for c in contributions if c["client_id"] == client_id]
        
        # Sort by round and timestamp
        contributions.sort(key=lambda x: (x["round"], x["timestamp"]))
        
        history = []
        for contrib in contributions:
            history_entry = {
                "round": contrib["round"],
                "client_id": contrib["client_id"],
                "timestamp": contrib["timestamp"],
                "num_samples": contrib["num_samples"],
                "withdrawn": contrib.get("withdrawn", False),
                "metrics": contrib.get("metrics", {}),
                "contribution_key": contrib.get("contribution_key"),
                "withdrawn_sample_ids": contrib.get("withdrawn_sample_ids", []),
            }
            history.append(history_entry)
        
        return history

