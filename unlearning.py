"""Machine unlearning algorithm for federated learning.
Implements incremental unlearning without retraining from scratch."""
import numpy as np
import torch
from typing import List, Tuple, Optional
from contributions_db import ContributionDB
from model import SimpleNet, get_parameters, set_parameters
from utils import load_data, test
import pickle
from pathlib import Path


class GradientBasedUnlearning:
    """
    Implements gradient-based machine unlearning for federated learning.
    
    This algorithm allows removing client contributions from the model
    without retraining from scratch by:
    1. Calculating the weighted difference between current model and 
       what it should be without the withdrawn contribution
    2. Applying an incremental update to "unlearn" the contribution
    3. Propagating the unlearning effect across subsequent rounds
    """
    
    def __init__(self, contribution_db: ContributionDB):
        """
        Initialize the unlearning algorithm.
        
        Args:
            contribution_db: ContributionDB instance for accessing contributions
        """
        self.db = contribution_db
    
    def unlearn_client_contribution(
        self,
        round_num: int,
        client_id: int,
        current_model_params: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Unlearn a specific client's contribution from a round.
        
        This implements the core unlearning formula:
        θ_unlearned = (θ_current * W_total - θ_withdrawn * w_withdrawn) / (W_total - w_withdrawn)
        
        Args:
            round_num: Round number where the contribution was made
            client_id: ID of the client to unlearn
            current_model_params: Current model parameters (if None, loads from saved aggregated model)
            
        Returns:
            Tuple of (unlearned_parameters, metadata_dict)
        """
        # Load current aggregated model if not provided
        if current_model_params is None:
            current_model_params = self._load_aggregated_model(round_num)
            if current_model_params is None:
                raise ValueError(f"Could not load aggregated model for round {round_num}")
        
        # Get all contributions for this round (including withdrawn)
        all_contributions = self.db.list_contributions(round_num=round_num, exclude_withdrawn=False)
        
        # Find the withdrawn client's contribution
        withdrawn_contrib = None
        total_weight = 0
        active_weight = 0
        
        for contrib in all_contributions:
            num_samples = contrib["num_samples"]
            total_weight += num_samples
            
            if contrib["client_id"] == client_id:
                withdrawn_contrib = contrib
            elif not contrib.get("withdrawn", False):
                active_weight += num_samples
        
        if withdrawn_contrib is None:
            raise ValueError(f"Client {client_id} did not contribute in round {round_num}")
        
        # Load withdrawn client's parameters
        round_dir = self.db.contributions_dir / f"round_{round_num:04d}"
        client_dir = round_dir / f"client_{client_id:04d}"
        params_file = client_dir / withdrawn_contrib["parameters_file"]
        
        if not params_file.exists():
            raise ValueError(f"Parameters file not found for client {client_id} in round {round_num}")
        
        with open(params_file, 'rb') as f:
            withdrawn_params = pickle.load(f)
        
        withdrawn_weight = withdrawn_contrib["num_samples"]
        
        # Calculate unlearned model using weighted subtraction
        # Formula: θ_unlearned = (θ_current * W_total - θ_withdrawn * w_withdrawn) / (W_total - w_withdrawn)
        unlearned_params = []
        
        for i in range(len(current_model_params)):
            # Weighted subtraction
            numerator = (
                np.array(current_model_params[i]) * total_weight - 
                np.array(withdrawn_params[i]) * withdrawn_weight
            )
            denominator = total_weight - withdrawn_weight
            
            if denominator <= 0:
                raise ValueError(f"Cannot unlearn: insufficient remaining weight (denominator={denominator})")
            
            unlearned_params.append(numerator / denominator)
        
        metadata = {
            "round": round_num,
            "client_id": client_id,
            "total_weight": total_weight,
            "withdrawn_weight": withdrawn_weight,
            "remaining_weight": total_weight - withdrawn_weight,
            "method": "gradient_based_unlearning"
        }
        
        return unlearned_params, metadata
    
    def unlearn_client_all_rounds(
        self,
        client_id: int,
        dataset_name: str = "MNIST",
        propagate: bool = True
    ) -> dict:
        """
        Unlearn a client's contributions across all rounds they participated in.
        
        This method:
        1. Identifies all rounds where the client contributed
        2. Unlearns the contribution in each round
        3. Optionally propagates the effect to subsequent rounds
        
        Args:
            client_id: ID of the client to unlearn
            dataset_name: Dataset name for model initialization
            propagate: If True, propagate unlearning effects to subsequent rounds
            
        Returns:
            Dictionary with unlearning results
        """
        # Get all rounds where this client contributed
        all_contributions = self.db.list_contributions(exclude_withdrawn=False)
        affected_rounds = sorted(set(
            contrib["round"] for contrib in all_contributions 
            if contrib["client_id"] == client_id
        ))
        
        if not affected_rounds:
            return {"error": f"Client {client_id} has no contributions"}
        
        results = {
            "client_id": client_id,
            "affected_rounds": affected_rounds,
            "unlearned_rounds": [],
            "propagated": propagate
        }
        
        # Unlearn in each affected round
        for round_num in affected_rounds:
            try:
                unlearned_params, metadata = self.unlearn_client_contribution(round_num, client_id)
                
                # Save unlearned model
                self.db.save_aggregated_model(
                    round_num=round_num,
                    parameters=unlearned_params,
                    metrics={
                        "unlearned": True,
                        "unlearned_client": client_id,
                        "method": "gradient_based_unlearning"
                    }
                )
                
                results["unlearned_rounds"].append({
                    "round": round_num,
                    "metadata": metadata,
                    "status": "success"
                })
                
            except Exception as e:
                results["unlearned_rounds"].append({
                    "round": round_num,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Propagate unlearning to subsequent rounds if requested
        if propagate and results["unlearned_rounds"]:
            self._propagate_unlearning(affected_rounds, dataset_name)
        
        return results
    
    def _propagate_unlearning(
        self,
        affected_rounds: List[int],
        dataset_name: str
    ):
        """
        Propagate unlearning effects to subsequent rounds.
        
        After unlearning in round R, subsequent rounds need to be updated
        because they were built on top of the model that included the withdrawn contribution.
        
        Args:
            affected_rounds: List of rounds where unlearning occurred
            dataset_name: Dataset name for model initialization
        """
        stats = self.db.get_statistics()
        all_rounds = sorted(stats["rounds"].keys())
        
        # Find rounds that come after the affected rounds
        max_affected = max(affected_rounds)
        subsequent_rounds = [r for r in all_rounds if r > max_affected]
        
        if not subsequent_rounds:
            return
        
        print(f"[Unlearning] Propagating unlearning effects to rounds: {subsequent_rounds}")
        
        # For each subsequent round, recalculate excluding withdrawn clients
        for round_num in subsequent_rounds:
            try:
                # Recalculate aggregated model excluding withdrawn clients
                recalculated_params = self.db.recalculate_aggregated_model(
                    round_num, 
                    exclude_withdrawn=True
                )
                
                if recalculated_params is not None:
                    # Save the recalculated model
                    self.db.save_aggregated_model(
                        round_num=round_num,
                        parameters=recalculated_params,
                        metrics={
                            "propagated_unlearning": True,
                            "method": "recalculation_after_unlearning"
                        }
                    )
                    print(f"[Unlearning] Propagated to round {round_num}")
            except Exception as e:
                print(f"[Unlearning] Failed to propagate to round {round_num}: {e}")
    
    def _load_aggregated_model(self, round_num: int) -> Optional[List[np.ndarray]]:
        """Load the aggregated model for a specific round."""
        round_dir = self.db.contributions_dir / f"round_{round_num:04d}"
        
        if not round_dir.exists():
            return None
        
        # Find the most recent aggregated model
        aggregated_files = list(round_dir.glob("round_*_aggregated_*_params.pkl"))
        if not aggregated_files:
            return None
        
        latest_file = max(aggregated_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'rb') as f:
            return pickle.load(f)
    
    def incremental_unlearn(
        self,
        round_num: int,
        withdrawn_client_ids: List[int],
        learning_rate: float = 1.0
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Incrementally unlearn multiple clients from a round using gradient-based approach.
        
        This is more efficient than unlearning clients one by one when multiple
        clients need to be unlearned from the same round.
        
        Args:
            round_num: Round number
            withdrawn_client_ids: List of client IDs to unlearn
            learning_rate: Learning rate for the unlearning update (default 1.0)
            
        Returns:
            Tuple of (unlearned_parameters, metadata_dict)
        """
        # Load current aggregated model
        current_params = self._load_aggregated_model(round_num)
        if current_params is None:
            raise ValueError(f"Could not load aggregated model for round {round_num}")
        
        # Get all contributions
        all_contributions = self.db.list_contributions(round_num=round_num, exclude_withdrawn=False)
        
        total_weight = sum(contrib["num_samples"] for contrib in all_contributions)
        withdrawn_weight = 0
        withdrawn_params_list = []
        
        # Collect all withdrawn contributions
        for contrib in all_contributions:
            if contrib["client_id"] in withdrawn_client_ids:
                withdrawn_weight += contrib["num_samples"]
                
                # Load parameters
                round_dir = self.db.contributions_dir / f"round_{round_num:04d}"
                client_dir = round_dir / f"client_{contrib['client_id']:04d}"
                params_file = client_dir / contrib["parameters_file"]
                
                if params_file.exists():
                    with open(params_file, 'rb') as f:
                        withdrawn_params_list.append((pickle.load(f), contrib["num_samples"]))
        
        if not withdrawn_params_list:
            # No contributions to unlearn
            return current_params, {"status": "no_withdrawn_contributions"}
        
        # Calculate weighted average of withdrawn contributions
        weighted_withdrawn = None
        for params, weight in withdrawn_params_list:
            if weighted_withdrawn is None:
                weighted_withdrawn = [p * weight for p in params]
            else:
                for i in range(len(weighted_withdrawn)):
                    weighted_withdrawn[i] += params[i] * weight
        
        # Normalize
        for i in range(len(weighted_withdrawn)):
            weighted_withdrawn[i] /= withdrawn_weight
        
        # Apply unlearning update
        remaining_weight = total_weight - withdrawn_weight
        if remaining_weight <= 0:
            raise ValueError("Cannot unlearn: no remaining contributions")
        
        unlearned_params = []
        for i in range(len(current_params)):
            # Unlearning formula with learning rate
            unlearned = (
                (np.array(current_params[i]) * total_weight - 
                 np.array(weighted_withdrawn[i]) * withdrawn_weight) / remaining_weight
            )
            # Apply learning rate for fine-tuning
            unlearned_params.append(
                current_params[i] + learning_rate * (unlearned - current_params[i])
            )
        
        metadata = {
            "round": round_num,
            "withdrawn_clients": withdrawn_client_ids,
            "total_weight": total_weight,
            "withdrawn_weight": withdrawn_weight,
            "remaining_weight": remaining_weight,
            "learning_rate": learning_rate,
            "method": "incremental_gradient_unlearning"
        }
        
        return unlearned_params, metadata


def apply_unlearning(
    dataset_name: str,
    client_id: int,
    propagate: bool = True
) -> dict:
    """
    Convenience function to apply unlearning to a withdrawn client.
    
    Args:
        dataset_name: Dataset name
        client_id: ID of the client to unlearn
        propagate: Whether to propagate unlearning to subsequent rounds
        
    Returns:
        Dictionary with unlearning results
    """
    db = ContributionDB(dataset_name=dataset_name)
    unlearner = GradientBasedUnlearning(db)
    
    return unlearner.unlearn_client_all_rounds(
        client_id=client_id,
        dataset_name=dataset_name,
        propagate=propagate
    )


class InfluenceFunctionBasedUnlearning:
    """
    Implements influence function-based machine unlearning for federated learning.
    
    This algorithm uses influence functions to estimate how much a client's
    contribution affected the model parameters, then removes that influence.
    
    Key differences from gradient-based approach:
    1. Estimates the "influence" of the contribution rather than direct subtraction
    2. Uses parameter differences to approximate gradients
    3. Applies influence-based correction with damping for stability
    4. Can handle cases where direct subtraction might be unstable
    """
    
    def __init__(self, contribution_db: ContributionDB, damping_factor: float = 0.01):
        """
        Initialize the influence function-based unlearning algorithm.
        
        Args:
            contribution_db: ContributionDB instance for accessing contributions
            damping_factor: Damping factor for numerical stability (default 0.01)
        """
        self.db = contribution_db
        self.damping_factor = damping_factor
    
    def _estimate_influence(
        self,
        current_params: List[np.ndarray],
        client_params: List[np.ndarray],
        client_weight: float,
        total_weight: float
    ) -> List[np.ndarray]:
        """
        Estimate the influence of a client's contribution on the model.
        
        The influence is estimated as the parameter difference weighted by
        the client's contribution weight relative to the total.
        
        Args:
            current_params: Current aggregated model parameters
            client_params: Client's model parameters
            client_weight: Weight (num_samples) of the client
            total_weight: Total weight of all clients
            
        Returns:
            Estimated influence as parameter differences
        """
        influence = []
        weight_ratio = client_weight / total_weight
        
        for i in range(len(current_params)):
            # Estimate influence as weighted parameter difference
            param_diff = np.array(client_params[i]) - np.array(current_params[i])
            # Scale by the client's relative contribution
            influence_i = param_diff * weight_ratio
            influence.append(influence_i)
        
        return influence
    
    def unlearn_client_contribution(
        self,
        round_num: int,
        client_id: int,
        current_model_params: Optional[List[np.ndarray]] = None,
        influence_scale: float = 1.0
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Unlearn a specific client's contribution using influence functions.
        
        This implements the influence-based unlearning formula:
        θ_unlearned = θ_current - I(θ_current, θ_client) * scale
        
        where I() is the estimated influence function.
        
        Args:
            round_num: Round number where the contribution was made
            client_id: ID of the client to unlearn
            current_model_params: Current model parameters (if None, loads from saved)
            influence_scale: Scaling factor for influence (default 1.0, can be tuned)
            
        Returns:
            Tuple of (unlearned_parameters, metadata_dict)
        """
        # Load current aggregated model if not provided
        if current_model_params is None:
            current_model_params = self._load_aggregated_model(round_num)
            if current_model_params is None:
                raise ValueError(f"Could not load aggregated model for round {round_num}")
        
        # Get all contributions for this round
        all_contributions = self.db.list_contributions(round_num=round_num, exclude_withdrawn=False)
        
        # Find the withdrawn client's contribution
        withdrawn_contrib = None
        total_weight = 0
        
        for contrib in all_contributions:
            num_samples = contrib["num_samples"]
            total_weight += num_samples
            
            if contrib["client_id"] == client_id:
                withdrawn_contrib = contrib
        
        if withdrawn_contrib is None:
            raise ValueError(f"Client {client_id} did not contribute in round {round_num}")
        
        # Load withdrawn client's parameters
        round_dir = self.db.contributions_dir / f"round_{round_num:04d}"
        client_dir = round_dir / f"client_{client_id:04d}"
        params_file = client_dir / withdrawn_contrib["parameters_file"]
        
        if not params_file.exists():
            raise ValueError(f"Parameters file not found for client {client_id} in round {round_num}")
        
        with open(params_file, 'rb') as f:
            withdrawn_params = pickle.load(f)
        
        withdrawn_weight = withdrawn_contrib["num_samples"]
        
        # Estimate the influence of the withdrawn client's contribution
        influence = self._estimate_influence(
            current_params=current_model_params,
            client_params=withdrawn_params,
            client_weight=withdrawn_weight,
            total_weight=total_weight
        )
        
        # Apply influence-based unlearning with damping for stability
        unlearned_params = []
        
        for i in range(len(current_model_params)):
            # Remove the estimated influence
            # Add damping to prevent numerical instability
            influence_corrected = influence[i] * influence_scale
            
            # Apply damping: reduce influence if it's too large relative to current params
            current_norm = np.linalg.norm(current_model_params[i])
            influence_norm = np.linalg.norm(influence_corrected)
            
            if current_norm > 0 and influence_norm > 0:
                # Adaptive damping: reduce influence if it's too large
                damping = min(1.0, (current_norm * self.damping_factor) / influence_norm)
                influence_corrected = influence_corrected * damping
            
            unlearned_param = np.array(current_model_params[i]) - influence_corrected
            unlearned_params.append(unlearned_param)
        
        metadata = {
            "round": round_num,
            "client_id": client_id,
            "total_weight": total_weight,
            "withdrawn_weight": withdrawn_weight,
            "influence_scale": influence_scale,
            "damping_factor": self.damping_factor,
            "method": "influence_function_based_unlearning"
        }
        
        return unlearned_params, metadata
    
    def unlearn_client_all_rounds(
        self,
        client_id: int,
        dataset_name: str = "MNIST",
        propagate: bool = True,
        influence_scale: float = 1.0
    ) -> dict:
        """
        Unlearn a client's contributions across all rounds using influence functions.
        
        Args:
            client_id: ID of the client to unlearn
            dataset_name: Dataset name for model initialization
            propagate: If True, propagate unlearning effects to subsequent rounds
            influence_scale: Scaling factor for influence estimation
            
        Returns:
            Dictionary with unlearning results
        """
        # Get all rounds where this client contributed
        all_contributions = self.db.list_contributions(exclude_withdrawn=False)
        affected_rounds = sorted(set(
            contrib["round"] for contrib in all_contributions 
            if contrib["client_id"] == client_id
        ))
        
        if not affected_rounds:
            return {"error": f"Client {client_id} has no contributions"}
        
        results = {
            "client_id": client_id,
            "affected_rounds": affected_rounds,
            "unlearned_rounds": [],
            "propagated": propagate,
            "method": "influence_function_based"
        }
        
        # Unlearn in each affected round
        for round_num in affected_rounds:
            try:
                unlearned_params, metadata = self.unlearn_client_contribution(
                    round_num, 
                    client_id,
                    influence_scale=influence_scale
                )
                
                # Save unlearned model
                self.db.save_aggregated_model(
                    round_num=round_num,
                    parameters=unlearned_params,
                    metrics={
                        "unlearned": True,
                        "unlearned_client": client_id,
                        "method": "influence_function_based_unlearning",
                        "influence_scale": influence_scale
                    }
                )
                
                results["unlearned_rounds"].append({
                    "round": round_num,
                    "metadata": metadata,
                    "status": "success"
                })
                
            except Exception as e:
                results["unlearned_rounds"].append({
                    "round": round_num,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Propagate unlearning to subsequent rounds if requested
        if propagate and results["unlearned_rounds"]:
            self._propagate_unlearning(affected_rounds, dataset_name)
        
        return results
    
    def _propagate_unlearning(
        self,
        affected_rounds: List[int],
        dataset_name: str
    ):
        """Propagate unlearning effects to subsequent rounds."""
        stats = self.db.get_statistics()
        all_rounds = sorted(stats["rounds"].keys())
        
        max_affected = max(affected_rounds)
        subsequent_rounds = [r for r in all_rounds if r > max_affected]
        
        if not subsequent_rounds:
            return
        
        print(f"[Influence Unlearning] Propagating to rounds: {subsequent_rounds}")
        
        for round_num in subsequent_rounds:
            try:
                recalculated_params = self.db.recalculate_aggregated_model(
                    round_num, 
                    exclude_withdrawn=True
                )
                
                if recalculated_params is not None:
                    self.db.save_aggregated_model(
                        round_num=round_num,
                        parameters=recalculated_params,
                        metrics={
                            "propagated_unlearning": True,
                            "method": "recalculation_after_influence_unlearning"
                        }
                    )
            except Exception as e:
                print(f"[Influence Unlearning] Failed to propagate to round {round_num}: {e}")
    
    def _load_aggregated_model(self, round_num: int) -> Optional[List[np.ndarray]]:
        """Load the aggregated model for a specific round."""
        round_dir = self.db.contributions_dir / f"round_{round_num:04d}"
        
        if not round_dir.exists():
            return None
        
        aggregated_files = list(round_dir.glob("round_*_aggregated_*_params.pkl"))
        if not aggregated_files:
            return None
        
        latest_file = max(aggregated_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'rb') as f:
            return pickle.load(f)


def evaluate_unlearned_model(
    dataset_name: str,
    round_num: int,
    unlearned_params: List[np.ndarray]
) -> dict:
    """
    Evaluate an unlearned model on test data.
    
    Args:
        dataset_name: Dataset name
        round_num: Round number
        unlearned_params: Unlearned model parameters
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dataset_name == "CIFAR10":
        net = SimpleNet(num_classes=10, num_channels=3, img_size=32).to(device)
    else:
        net = SimpleNet(num_classes=10, num_channels=1, img_size=28).to(device)
    
    _, testloader = load_data(dataset_name, num_clients=1, batch_size=32)
    
    set_parameters(net, unlearned_params)
    loss, accuracy = test(net, testloader, device)
    
    return {
        "round": round_num,
        "loss": float(loss),
        "accuracy": float(accuracy)
    }

