"""Contribution database for tracking client contributions in federated learning."""
import os
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle


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
        
        # Save metadata
        metadata = {
            "round": round_num,
            "client_id": client_id,
            "timestamp": timestamp,
            "num_samples": num_samples,
            "parameters_count": len(parameters),
            "metrics": metrics or {},
            "parameters_file": f"{base_filename}_params.pkl",
            "withdrawn": is_withdrawn
        }
        
        metadata_file = client_dir / f"{base_filename}_metadata.json"
        with open(metadata_file, 'w') as f:
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
    
    def get_contribution_info(self, round_num, client_id):
        """
        Retrieve information about a specific contribution.
        
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
        with open(latest_metadata, 'r') as f:
            return json.load(f)
    
    def list_contributions(self, round_num=None):
        """
        List all contributions, optionally filtered by round.
        
        Args:
            round_num: Optional round number to filter by
            
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
                        with open(metadata_file, 'r') as f:
                            contributions.append(json.load(f))
        
        return contributions
    
    def get_statistics(self):
        """Get overall statistics about contributions."""
        stats = self.stats.copy()
        stats["clients_participated"] = list(stats["clients_participated"])
        # Convert sets in rounds
        for round_num in stats["rounds"]:
            stats["rounds"][round_num]["clients"] = list(stats["rounds"][round_num]["clients"])
        return stats
    
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
                        with open(metadata_file, 'r') as f:
                            contrib = json.load(f)
                        
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
            
            num_samples = contrib["num_samples"]
            
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
                "metrics": contrib.get("metrics", {})
            }
            history.append(history_entry)
        
        return history

