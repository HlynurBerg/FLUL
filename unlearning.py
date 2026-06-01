"""Machine unlearning algorithm for federated learning.
Implements incremental unlearning without retraining from scratch."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from typing import Dict, List, Optional, Tuple
from contributions_db import ContributionDB
from model import DEFAULT_DATASET_CONFIGS, SimpleNet, get_parameters, set_parameters
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

    The influence of sample s on the global aggregate is estimated as:

        I_s = (θ_c − θ_incoming) × w_s × (n_c / N_total)

    where:
      θ_c        — client model after local training this round
      θ_incoming — global model the client received before training (not θ_agg)
      w_s        — scalar weight for sample s:
                     grad_norm_s / Σ_j grad_norm_j   (if training_trace has per_sample_grad_norm)
                     loss_s       / Σ_j loss_j         (fallback: per_sample_loss)
                     1 / n_c                            (uniform fallback)
      n_c        — number of samples used by this client
      N_total    — total samples across all clients this round

    The unlearned model is:
        θ_unlearned = θ_agg − I_s × scale

    Key difference from the old approach: the parameter difference uses
    θ_c − θ_incoming (the actual local training update) rather than
    θ_c − θ_agg (the client-vs-consensus deviation that cannot be decomposed
    per-sample). The per-sample gradient norms decompose that training update
    into per-sample contributions.
    """

    def __init__(self, contribution_db: ContributionDB, damping_factor: float = 0.01):
        """
        Args:
            contribution_db: ContributionDB instance for accessing contributions
            damping_factor: Damping factor for adaptive clipping (default 0.01)
        """
        self.db = contribution_db
        self.damping_factor = damping_factor

    # ------------------------------------------------------------------
    # Core influence estimation
    # ------------------------------------------------------------------

    def _estimate_per_sample_influence(
        self,
        incoming_params: List[np.ndarray],
        client_params: List[np.ndarray],
        client_weight: float,
        total_weight: float,
        sample_weight_fraction: float,
    ) -> List[np.ndarray]:
        """
        Estimate the influence of a subset of samples on the global aggregate.

        Args:
            incoming_params: Global model weights the client received (θ_incoming)
            client_params: Client's trained model weights (θ_c)
            client_weight: n_c — samples used by this client
            total_weight: N_total — total samples across all clients this round
            sample_weight_fraction: Scalar in [0, 1] — fraction of the client's
                gradient mass attributable to the target sample(s). Use 1.0 to
                unlearn the full client contribution.

        Returns:
            Per-layer influence arrays (same shapes as model parameters)
        """
        weight_ratio = client_weight / total_weight
        influence = []
        for inc, cli in zip(incoming_params, client_params):
            # Local training update for this layer
            client_delta = np.array(cli) - np.array(inc)
            # Fraction of that update attributable to the target samples,
            # scaled by the client's weight in the FedAvg aggregate
            influence.append(client_delta * sample_weight_fraction * weight_ratio)
        return influence

    def _compute_sample_weight_fraction(
        self,
        contribution_metrics: dict,
        target_sample_ids: List[int],
    ) -> float:
        """
        Compute the fraction of total gradient mass attributable to target_sample_ids.

        Priority:
        1. per_sample_grad_norm  — most accurate (requires --training-trace full)
        2. per_sample_loss       — proxy (requires --training-trace loss or full)
        3. uniform               — 1 / n_client_samples

        Args:
            contribution_metrics: The ``metrics`` dict from contribution metadata
                (already merged with the sample manifest by ContributionDB).
            target_sample_ids: Global sample IDs whose influence to estimate.

        Returns:
            Scalar weight fraction in [0, 1].
        """
        target_set = set(int(s) for s in target_sample_ids)
        training_trace = contribution_metrics.get("training_trace") or {}

        # 1. Per-sample gradient norms (list of {sample_id, grad_norm})
        grad_norms = training_trace.get("per_sample_grad_norm")
        if grad_norms:
            total = sum(r["grad_norm"] for r in grad_norms)
            if total > 0:
                withdrawn = sum(
                    r["grad_norm"] for r in grad_norms
                    if int(r["sample_id"]) in target_set
                )
                return withdrawn / total

        # 2. Per-sample losses as a proxy for gradient magnitude
        per_sample_loss = training_trace.get("per_sample_loss")
        if per_sample_loss:
            total = sum(r["loss"] for r in per_sample_loss)
            if total > 0:
                withdrawn = sum(
                    r["loss"] for r in per_sample_loss
                    if int(r["sample_id"]) in target_set
                )
                return withdrawn / total

        # 3. Uniform fallback: each sample has equal influence
        all_sample_ids = contribution_metrics.get("sample_ids") or []
        n_total = len(all_sample_ids) if all_sample_ids else max(
            contribution_metrics.get("num_samples", 1), 1
        )
        return len(target_sample_ids) / n_total

    def _apply_damped_influence(
        self,
        current_model_params: List[np.ndarray],
        influence: List[np.ndarray],
        influence_scale: float,
    ) -> List[np.ndarray]:
        """Subtract influence from current params with adaptive damping."""
        unlearned_params = []
        for cur, inf in zip(current_model_params, influence):
            inf_scaled = inf * influence_scale
            current_norm = np.linalg.norm(cur)
            inf_norm = np.linalg.norm(inf_scaled)
            if current_norm > 0 and inf_norm > 0:
                damping = min(1.0, (current_norm * self.damping_factor) / inf_norm)
                inf_scaled = inf_scaled * damping
            unlearned_params.append(np.array(cur) - inf_scaled)
        return unlearned_params

    def _load_contribution_data(self, round_num: int, client_id: int):
        """Load contrib metadata, client params, and θ_incoming for a round/client."""
        contrib = self.db.get_contribution_info(round_num, client_id)
        if contrib is None:
            raise ValueError(f"No contribution found for round {round_num}, client {client_id}")

        round_dir = self.db.contributions_dir / f"round_{round_num:04d}"
        client_dir = round_dir / f"client_{client_id:04d}"
        params_file = client_dir / contrib["parameters_file"]
        if not params_file.exists():
            raise ValueError(
                f"Parameters file not found for client {client_id} in round {round_num}"
            )
        with open(params_file, "rb") as f:
            client_params = pickle.load(f)

        incoming_params = self.db.load_incoming_global_for_round(round_num)
        if incoming_params is None:
            raise ValueError(
                f"Cannot load incoming global model for round {round_num}. "
                "Ensure initial_global_params.pkl (round 0) or the previous round's "
                "aggregated checkpoint exists."
            )

        return contrib, client_params, incoming_params

    # ------------------------------------------------------------------
    # Public unlearning methods
    # ------------------------------------------------------------------

    def unlearn_samples_from_contribution(
        self,
        round_num: int,
        client_id: int,
        sample_ids: List[int],
        current_model_params: Optional[List[np.ndarray]] = None,
        influence_scale: float = 1.0,
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Remove the influence of specific samples from one client's contribution.

        Uses stored per-sample gradient norms (or losses) from the training trace
        to estimate each sample's share of the client's gradient update, then
        subtracts that share's contribution from the global aggregate.

        Args:
            round_num: Round in which these samples were used for training
            client_id: Client that trained on the samples
            sample_ids: Global sample IDs to unlearn
            current_model_params: θ_agg to correct (loaded from disk if None)
            influence_scale: Multiplier applied to the influence before subtraction

        Returns:
            (unlearned_parameters, metadata_dict)
        """
        if not sample_ids:
            raise ValueError("sample_ids must be non-empty")

        if current_model_params is None:
            current_model_params = self._load_aggregated_model(round_num)
            if current_model_params is None:
                raise ValueError(f"Could not load aggregated model for round {round_num}")

        contrib, client_params, incoming_params = self._load_contribution_data(
            round_num, client_id
        )

        all_contributions = self.db.list_contributions(
            round_num=round_num, exclude_withdrawn=False
        )
        total_weight = sum(c["num_samples"] for c in all_contributions)
        client_weight = contrib["num_samples"]

        metrics = contrib.get("metrics") or {}
        sample_weight_fraction = self._compute_sample_weight_fraction(metrics, sample_ids)

        influence = self._estimate_per_sample_influence(
            incoming_params=incoming_params,
            client_params=client_params,
            client_weight=client_weight,
            total_weight=total_weight,
            sample_weight_fraction=sample_weight_fraction,
        )

        unlearned_params = self._apply_damped_influence(
            current_model_params, influence, influence_scale
        )

        # Determine which weighting source was used for transparency
        training_trace = metrics.get("training_trace") or {}
        if training_trace.get("per_sample_grad_norm"):
            weight_source = "per_sample_grad_norm"
        elif training_trace.get("per_sample_loss"):
            weight_source = "per_sample_loss"
        else:
            weight_source = "uniform"

        metadata = {
            "round": round_num,
            "client_id": client_id,
            "sample_ids": sorted(int(s) for s in sample_ids),
            "total_weight": total_weight,
            "client_weight": client_weight,
            "sample_weight_fraction": sample_weight_fraction,
            "weight_source": weight_source,
            "influence_scale": influence_scale,
            "damping_factor": self.damping_factor,
            "method": "per_sample_influence_unlearning",
        }
        return unlearned_params, metadata

    def unlearn_withdrawn_samples(
        self,
        round_num: int,
        client_id: int,
        current_model_params: Optional[List[np.ndarray]] = None,
        influence_scale: float = 1.0,
        contribution_key: Optional[str] = None,
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Convenience wrapper: read withdrawn sample IDs from the database and
        call ``unlearn_samples_from_contribution``.

        Raises ``ValueError`` if no sample withdrawals are active for this
        contribution.
        """
        ck = self.db.resolve_contribution_key(round_num, client_id, contribution_key)
        state = self.db.get_sample_withdrawal_state(ck)
        if not state:
            raise ValueError(
                f"No active sample withdrawals for contribution {ck}"
            )
        sample_ids = [int(s) for s in state["sample_ids"]]
        return self.unlearn_samples_from_contribution(
            round_num=round_num,
            client_id=client_id,
            sample_ids=sample_ids,
            current_model_params=current_model_params,
            influence_scale=influence_scale,
        )

    def unlearn_client_contribution(
        self,
        round_num: int,
        client_id: int,
        current_model_params: Optional[List[np.ndarray]] = None,
        influence_scale: float = 1.0,
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Unlearn an entire client's contribution for one round.

        Equivalent to ``unlearn_samples_from_contribution`` with
        ``sample_weight_fraction = 1.0`` (all samples are removed).  The
        training-update baseline is θ_c − θ_incoming, NOT θ_c − θ_agg, so
        the subtracted quantity is the client's actual gradient contribution
        to FedAvg rather than its deviation from the consensus.

        Args:
            round_num: Round number where the contribution was made
            client_id: ID of the client to unlearn
            current_model_params: Current aggregated parameters (loaded if None)
            influence_scale: Scaling factor for influence (default 1.0)

        Returns:
            Tuple of (unlearned_parameters, metadata_dict)
        """
        if current_model_params is None:
            current_model_params = self._load_aggregated_model(round_num)
            if current_model_params is None:
                raise ValueError(f"Could not load aggregated model for round {round_num}")

        contrib, client_params, incoming_params = self._load_contribution_data(
            round_num, client_id
        )

        all_contributions = self.db.list_contributions(
            round_num=round_num, exclude_withdrawn=False
        )
        total_weight = sum(c["num_samples"] for c in all_contributions)
        client_weight = contrib["num_samples"]

        # All samples from this client are being removed → fraction = 1.0
        influence = self._estimate_per_sample_influence(
            incoming_params=incoming_params,
            client_params=client_params,
            client_weight=client_weight,
            total_weight=total_weight,
            sample_weight_fraction=1.0,
        )

        unlearned_params = self._apply_damped_influence(
            current_model_params, influence, influence_scale
        )

        metadata = {
            "round": round_num,
            "client_id": client_id,
            "total_weight": total_weight,
            "withdrawn_weight": client_weight,
            "influence_scale": influence_scale,
            "damping_factor": self.damping_factor,
            "method": "influence_function_based_unlearning",
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

    def unlearn_samples_all_rounds(
        self,
        client_id: int,
        sample_ids: List[int],
        dataset_name: str = "MNIST",
        propagate: bool = True,
        influence_scale: float = 1.0,
    ) -> dict:
        """
        Unlearn specific samples from a client's contributions across all rounds.

        Finds every round where ``sample_ids`` appear in the client's training
        trace and applies per-sample influence unlearning to each one.  If no
        sample manifest is stored for a round, all target samples are assumed
        present (conservative fallback).

        Args:
            client_id: ID of the client that trained on the samples.
            sample_ids: Global sample IDs to remove.
            dataset_name: Dataset name (used for propagation metadata).
            propagate: If True, recalculate aggregated models for subsequent
                       rounds using the corrected per-sample weights.
            influence_scale: Scaling factor passed to ``unlearn_samples_from_contribution``.

        Returns:
            Dictionary with keys ``client_id``, ``sample_ids``,
            ``sample_rounds`` (mapping round → found IDs),
            ``unlearned_rounds`` (list of per-round results), ``propagated``.
        """
        if not sample_ids:
            raise ValueError("sample_ids must be non-empty")

        sample_rounds = self.db.find_rounds_containing_samples(client_id, sample_ids)

        results = {
            "client_id": client_id,
            "sample_ids": sorted(int(s) for s in sample_ids),
            "sample_rounds": sample_rounds,
            "unlearned_rounds": [],
            "propagated": propagate,
            "method": "per_sample_influence_unlearning",
        }

        if not sample_rounds:
            results["error"] = (
                f"Client {client_id} has no contributions containing the requested samples"
            )
            return results

        for round_num in sorted(sample_rounds.keys()):
            round_sample_ids = sample_rounds[round_num]
            try:
                unlearned_params, metadata = self.unlearn_samples_from_contribution(
                    round_num=round_num,
                    client_id=client_id,
                    sample_ids=round_sample_ids,
                    influence_scale=influence_scale,
                )
                self.db.save_aggregated_model(
                    round_num=round_num,
                    parameters=unlearned_params,
                    metrics={
                        "unlearned_samples": True,
                        "unlearned_client": client_id,
                        "unlearned_sample_ids": metadata["sample_ids"],
                        "method": "per_sample_influence_unlearning",
                        "weight_source": metadata["weight_source"],
                        "influence_scale": influence_scale,
                    },
                )
                results["unlearned_rounds"].append({
                    "round": round_num,
                    "metadata": metadata,
                    "status": "success",
                })
            except Exception as e:
                results["unlearned_rounds"].append({
                    "round": round_num,
                    "status": "failed",
                    "error": str(e),
                })

        if propagate and results["unlearned_rounds"]:
            self._propagate_sample_unlearning(
                from_round=max(sample_rounds.keys()),
                dataset_name=dataset_name,
            )

        return results

    def _propagate_sample_unlearning(self, from_round: int, dataset_name: str) -> List[int]:
        """
        Propagate per-sample unlearning to rounds STRICTLY AFTER ``from_round``.

        Recalculates aggregated models for subsequent rounds using
        ``recalculate_aggregated_model(exclude_withdrawn=True)``, which now
        accounts for per-sample withdrawals by reducing fed-averaging weights.

        Args:
            from_round: LATEST round where per-sample unlearning was already
                applied (typically ``max(sample_rounds.keys())``). Pass the
                latest, not the earliest, so per-round aggregate corrections
                made by ``unlearn_samples_from_contribution`` are not
                overwritten by FedAvg recalculation.
            dataset_name: Dataset name (for logging).

        Returns:
            List of round numbers that were recalculated.
        """
        stats = self.db.get_statistics()
        all_rounds = sorted(stats["rounds"].keys())
        subsequent_rounds = [r for r in all_rounds if r > from_round]

        if not subsequent_rounds:
            return []

        print(f"[Influence Unlearning] Propagating sample unlearning to rounds: {subsequent_rounds}")
        updated = []

        for round_num in subsequent_rounds:
            try:
                recalculated_params = self.db.recalculate_aggregated_model(
                    round_num, exclude_withdrawn=True
                )
                if recalculated_params is not None:
                    self.db.save_aggregated_model(
                        round_num=round_num,
                        parameters=recalculated_params,
                        metrics={
                            "propagated_sample_unlearning": True,
                            "method": "recalculation_after_sample_unlearning",
                        },
                    )
                    updated.append(round_num)
                    print(f"[Influence Unlearning] Propagated sample unlearning to round {round_num}")
            except Exception as e:
                print(f"[Influence Unlearning] Failed to propagate sample unlearning to round {round_num}: {e}")

        return updated

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


class HessianInfluenceUnlearning:
    """
    Hessian-based machine unlearning using inverse Hessian-vector products (IHVP).

    Computes the second-order influence function correction:
        θ_new = θ_agg − H_{θ_incoming}⁻¹ · ∇L_forget(θ_incoming) · scale

    where H is the Hessian of the client's training loss, approximated iteratively
    via conjugate gradient (CG) using PyTorch's autograd for Hessian-vector products
    (the "Pearlmutter trick": two backward passes, ``create_graph=True``).

    This is strictly more accurate than ``InfluenceFunctionBasedUnlearning`` because it
    accounts for the curvature of the loss landscape rather than treating the parameter
    space as flat.

    Args:
        contribution_db: ContributionDB for loading contributions and checkpoints.
        model: A PyTorch model instance whose architecture matches the saved checkpoints.
               Its parameters will be overwritten during computation.
        dataset_name: Dataset name passed to ``load_data`` to reconstruct client partitions.
        num_clients: Total number of clients used during training; needed to reconstruct
                     the exact data partition for each client_id.
        device: Torch device. Defaults to CUDA if available, else CPU.
        damping: Regularisation added to the Hessian diagonal (H + λI) to ensure positive
                 definiteness and stable CG convergence. Larger values → smaller, safer steps.
        cg_max_iter: Maximum CG iterations. More → more accurate IHVP, but slower.
        cg_tol: CG convergence threshold on the residual norm.
        batch_size: Batch size for client data loaders used during gradient/HVP computation.
    """

    def __init__(
        self,
        contribution_db: ContributionDB,
        model: torch.nn.Module,
        dataset_name: str,
        num_clients: int,
        device: Optional[torch.device] = None,
        damping: float = 0.1,
        cg_max_iter: int = 50,
        cg_tol: float = 1e-4,
        batch_size: int = 32,
    ):
        self.db = contribution_db
        self.model = model
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.damping = damping
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_client_dataset(self, client_id: int):
        """Return the WithSampleIds dataset for client_id using the stored partition."""
        # Strip any _CLASS_VERTICAL or similar suffix for load_data compatibility
        base_dataset = self.dataset_name.split("_")[0] if "_" in self.dataset_name else self.dataset_name
        client_loaders, _ = load_data(base_dataset, num_clients=self.num_clients, batch_size=self.batch_size)
        return client_loaders[client_id].dataset

    def _make_loader(self, dataset, sample_ids: Optional[List[int]] = None) -> DataLoader:
        """
        Build a DataLoader over ``dataset`` (a WithSampleIds instance).

        If ``sample_ids`` is given, only samples whose global_id is in ``sample_ids``
        are included (via a Subset). The loader yields (image, label, global_id) triples.
        """
        if sample_ids is None:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        target = set(int(s) for s in sample_ids)
        indices = [i for i, gid in enumerate(dataset.global_ids) if gid in target]
        return DataLoader(Subset(dataset, indices), batch_size=self.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Core second-order computation
    # ------------------------------------------------------------------

    def _compute_forget_gradient(self, forget_loader: DataLoader) -> torch.Tensor:
        """
        Compute the mean gradient of the cross-entropy loss over the forget samples.

        The model must already be set to the desired parameter point (θ_incoming)
        before calling this method.

        Returns a flat 1-D tensor of shape (n_params,).
        """
        self.model.eval()
        self.model.zero_grad()
        n_batches = 0
        for batch in forget_loader:
            images, labels, _ = batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            # Divide inside the loop so each batch contributes 1/n_batches to the total
            loss = F.cross_entropy(outputs, labels) / len(forget_loader)
            loss.backward()
            n_batches += 1
        if n_batches == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            return torch.zeros(n_params, device=self.device)
        return torch.cat([
            p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=self.device)
            for p in self.model.parameters()
        ]).detach()

    def _hvp(self, retain_loader: DataLoader, v: torch.Tensor) -> torch.Tensor:
        """
        Compute (H + damping·I) @ v using the Pearlmutter trick.

        For each mini-batch:
          1. Forward pass → loss
          2. Compute gradients with create_graph=True  (keeps the computation graph)
          3. Differentiate the dot product (flat_grad · v) w.r.t. parameters → H @ v

        Averages HVPs over all batches and adds the damping term.

        Args:
            retain_loader: DataLoader over the retain (Hessian) dataset.
            v: Flat parameter-space vector to multiply by H.

        Returns:
            Flat tensor (H + λI) @ v.
        """
        self.model.eval()
        v = v.to(self.device).detach()
        total_hvp = torch.zeros_like(v)
        n_batches = 0
        for batch in retain_loader:
            images, labels, _ = batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)
            # First backward: gradient with computation graph retained
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            flat_grad = torch.cat([g.flatten() for g in grads])
            # Second backward: gradient of (flat_grad · v) → H @ v
            hvp_grads = torch.autograd.grad(
                (flat_grad * v).sum(),
                self.model.parameters(),
                retain_graph=False,
            )
            hvp = torch.cat([h.flatten() for h in hvp_grads])
            total_hvp = total_hvp + hvp.detach()
            n_batches += 1
        if n_batches == 0:
            return self.damping * v
        return total_hvp / n_batches + self.damping * v

    def _cg_solve(self, retain_loader: DataLoader, g: torch.Tensor) -> torch.Tensor:
        """
        Solve H @ x = g for x via conjugate gradient, using ``_hvp`` for matrix-vector products.

        Convergence is declared when the residual norm drops below ``cg_tol`` or
        after ``cg_max_iter`` iterations.

        Returns the flat solution vector x ≈ H⁻¹ g.
        """
        g = g.to(self.device)
        x = torch.zeros_like(g)
        r = g.clone()
        p = g.clone()
        rs_old = torch.dot(r, r)

        for i in range(self.cg_max_iter):
            Ap = self._hvp(retain_loader, p)
            pAp = torch.dot(p, Ap)
            if pAp.abs() < 1e-12:
                break
            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if rs_new.sqrt() < self.cg_tol:
                break
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

        return x

    # ------------------------------------------------------------------
    # Parameter conversion helpers
    # ------------------------------------------------------------------

    def _set_model_params(self, params: List[np.ndarray]):
        """Load numpy parameter arrays into the model."""
        set_parameters(self.model, params)
        self.model.to(self.device)

    def _ihvp_to_param_arrays(self, ihvp: torch.Tensor, reference_params: List[np.ndarray]) -> List[np.ndarray]:
        """Split flat IHVP tensor back into per-state_dict-entry numpy arrays.

        ``ihvp`` is flat over ``self.model.parameters()`` (learnable params only),
        but ``reference_params`` mirrors the full ``state_dict`` order — which
        also contains buffers (BatchNorm's ``running_mean`` / ``running_var`` /
        ``num_batches_tracked``). Buffers have no gradient and therefore no
        IHVP entry, so we emit zero corrections for them so the subsequent
        elementwise subtraction is a no-op on those slots.
        """
        # Names of learnable parameters in state_dict order (parameters come
        # before buffers within each module, so the relative order in
        # state_dict matches named_parameters()).
        param_names = {name for name, _ in self.model.named_parameters()}
        sd_keys = list(self.model.state_dict().keys())
        if len(sd_keys) != len(reference_params):
            raise RuntimeError(
                f"reference_params length ({len(reference_params)}) does not match "
                f"model.state_dict() length ({len(sd_keys)}). Did the model "
                f"architecture diverge from the saved checkpoint?"
            )

        result = []
        offset = 0
        for key, ref in zip(sd_keys, reference_params):
            if key in param_names:
                numel = ref.size
                layer = ihvp[offset:offset + numel].cpu().numpy().reshape(ref.shape)
                offset += numel
            else:
                # Buffer (e.g. BatchNorm running stats): no IHVP correction.
                layer = np.zeros_like(ref)
            result.append(layer)
        if offset != ihvp.numel():
            raise RuntimeError(
                f"IHVP slicing left {ihvp.numel() - offset} unconsumed elements; "
                f"state_dict and named_parameters() disagree on parameter ordering."
            )
        return result

    # ------------------------------------------------------------------
    # Public unlearning methods
    # ------------------------------------------------------------------

    def unlearn_samples_from_contribution(
        self,
        round_num: int,
        client_id: int,
        sample_ids: List[int],
        current_model_params: Optional[List[np.ndarray]] = None,
        influence_scale: float = 1.0,
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Remove the influence of specific samples using the second-order IHVP correction.

        Steps:
          1. Load θ_agg and θ_incoming.
          2. Set model to θ_incoming.
          3. Compute ∇L_forget at θ_incoming over the forget samples.
          4. Compute H⁻¹ · ∇L_forget via CG using the retain dataset as the Hessian source.
          5. Scale the IHVP by (n_forget / n_total) · influence_scale and subtract from θ_agg.

        Args:
            round_num: Round in which these samples were used for training.
            client_id: Client that trained on the samples.
            sample_ids: Global sample IDs to unlearn.
            current_model_params: θ_agg to correct (loaded from disk if None).
            influence_scale: Additional multiplier on the IHVP correction.

        Returns:
            (unlearned_parameters, metadata_dict)
        """
        if not sample_ids:
            raise ValueError("sample_ids must be non-empty")

        if current_model_params is None:
            current_model_params = self._load_aggregated_model(round_num)
            if current_model_params is None:
                raise ValueError(f"Could not load aggregated model for round {round_num}")

        incoming_params = self.db.load_incoming_global_for_round(round_num)
        if incoming_params is None:
            raise ValueError(
                f"Cannot load incoming global model for round {round_num}. "
                "Ensure initial_global_params.pkl (round 0) or the previous round's "
                "aggregated checkpoint exists."
            )

        # FL weight information for scaling
        all_contributions = self.db.list_contributions(round_num=round_num, exclude_withdrawn=False)
        total_weight = sum(c["num_samples"] for c in all_contributions)
        client_weight = next(
            (c["num_samples"] for c in all_contributions if c["client_id"] == client_id), 0
        )

        # Set model to θ_incoming for gradient/Hessian computation
        self._set_model_params(incoming_params)

        # Build data loaders
        client_dataset = self._get_client_dataset(client_id)
        forget_set = set(int(s) for s in sample_ids)
        retain_ids = [gid for gid in client_dataset.global_ids if gid not in forget_set]

        forget_loader = self._make_loader(client_dataset, list(forget_set))
        # Use retain samples for the Hessian; fall back to full client data if all are forgotten
        retain_loader = self._make_loader(client_dataset, retain_ids) if retain_ids else self._make_loader(client_dataset)

        # Compute forget gradient at θ_incoming
        g_forget = self._compute_forget_gradient(forget_loader)

        # Compute IHVP: H_retain⁻¹ · g_forget
        ihvp = self._cg_solve(retain_loader, g_forget)

        # Scale: (n_forget / n_total) captures the FL weight of this forget set
        n_forget = len(sample_ids)
        scale = (n_forget / max(total_weight, 1)) * influence_scale
        ihvp_scaled = ihvp * scale

        # Apply correction: θ_new = θ_agg - H⁻¹ g · scale
        ihvp_arrays = self._ihvp_to_param_arrays(ihvp_scaled, current_model_params)
        unlearned_params = [
            np.array(cur) - np.array(upd)
            for cur, upd in zip(current_model_params, ihvp_arrays)
        ]

        metadata = {
            "round": round_num,
            "client_id": client_id,
            "sample_ids": sorted(int(s) for s in sample_ids),
            "total_weight": total_weight,
            "client_weight": client_weight,
            "n_forget": n_forget,
            "fl_scale": float(n_forget / max(total_weight, 1)),
            "influence_scale": influence_scale,
            "damping": self.damping,
            "cg_max_iter": self.cg_max_iter,
            "cg_tol": self.cg_tol,
            "method": "hessian_ihvp_unlearning",
        }
        return unlearned_params, metadata

    def unlearn_client_contribution(
        self,
        round_num: int,
        client_id: int,
        current_model_params: Optional[List[np.ndarray]] = None,
        influence_scale: float = 1.0,
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Unlearn an entire client's contribution for one round.

        The forget gradient is computed over all of the client's training samples.
        The Hessian is also computed over all client samples (since there is no retain set).

        The FL scale factor is (client_weight / total_weight).
        """
        if current_model_params is None:
            current_model_params = self._load_aggregated_model(round_num)
            if current_model_params is None:
                raise ValueError(f"Could not load aggregated model for round {round_num}")

        incoming_params = self.db.load_incoming_global_for_round(round_num)
        if incoming_params is None:
            raise ValueError(
                f"Cannot load incoming global model for round {round_num}. "
                "Ensure initial_global_params.pkl (round 0) or the previous round's "
                "aggregated checkpoint exists."
            )

        all_contributions = self.db.list_contributions(round_num=round_num, exclude_withdrawn=False)
        total_weight = sum(c["num_samples"] for c in all_contributions)
        client_weight = next(
            (c["num_samples"] for c in all_contributions if c["client_id"] == client_id), 0
        )

        self._set_model_params(incoming_params)

        client_dataset = self._get_client_dataset(client_id)
        full_loader = self._make_loader(client_dataset)

        g_forget = self._compute_forget_gradient(full_loader)
        ihvp = self._cg_solve(full_loader, g_forget)

        scale = (client_weight / max(total_weight, 1)) * influence_scale
        ihvp_scaled = ihvp * scale

        ihvp_arrays = self._ihvp_to_param_arrays(ihvp_scaled, current_model_params)
        unlearned_params = [
            np.array(cur) - np.array(upd)
            for cur, upd in zip(current_model_params, ihvp_arrays)
        ]

        metadata = {
            "round": round_num,
            "client_id": client_id,
            "total_weight": total_weight,
            "client_weight": client_weight,
            "fl_scale": float(client_weight / max(total_weight, 1)),
            "influence_scale": influence_scale,
            "damping": self.damping,
            "cg_max_iter": self.cg_max_iter,
            "cg_tol": self.cg_tol,
            "method": "hessian_ihvp_unlearning",
        }
        return unlearned_params, metadata

    def unlearn_client_all_rounds(
        self,
        client_id: int,
        dataset_name: str = "MNIST",
        propagate: bool = True,
        influence_scale: float = 1.0,
    ) -> dict:
        """
        Unlearn a client's contributions across all rounds using Hessian IHVP.

        Args:
            client_id: ID of the client to unlearn.
            dataset_name: Dataset name for model initialization (used during propagation).
            propagate: If True, propagate unlearning effects to subsequent rounds.
            influence_scale: Additional multiplier on the IHVP correction.

        Returns:
            Dictionary with unlearning results.
        """
        all_contributions = self.db.list_contributions(exclude_withdrawn=False)
        affected_rounds = sorted(set(
            c["round"] for c in all_contributions if c["client_id"] == client_id
        ))

        if not affected_rounds:
            return {"error": f"Client {client_id} has no contributions"}

        results = {
            "client_id": client_id,
            "affected_rounds": affected_rounds,
            "unlearned_rounds": [],
            "propagated": propagate,
            "method": "hessian_ihvp",
        }

        for round_num in affected_rounds:
            try:
                unlearned_params, metadata = self.unlearn_client_contribution(
                    round_num, client_id, influence_scale=influence_scale
                )
                self.db.save_aggregated_model(
                    round_num=round_num,
                    parameters=unlearned_params,
                    metrics={
                        "unlearned": True,
                        "unlearned_client": client_id,
                        "method": "hessian_ihvp_unlearning",
                        "influence_scale": influence_scale,
                    },
                )
                results["unlearned_rounds"].append({
                    "round": round_num,
                    "metadata": metadata,
                    "status": "success",
                })
            except Exception as e:
                results["unlearned_rounds"].append({
                    "round": round_num,
                    "status": "failed",
                    "error": str(e),
                })

        if propagate and results["unlearned_rounds"]:
            self._propagate_unlearning(affected_rounds, dataset_name)

        return results

    def _propagate_unlearning(self, affected_rounds: List[int], dataset_name: str):
        """Propagate unlearning effects to subsequent rounds (same pattern as other unlearners)."""
        stats = self.db.get_statistics()
        all_rounds = sorted(stats["rounds"].keys())
        max_affected = max(affected_rounds)
        subsequent_rounds = [r for r in all_rounds if r > max_affected]
        if not subsequent_rounds:
            return
        print(f"[Hessian Unlearning] Propagating to rounds: {subsequent_rounds}")
        for round_num in subsequent_rounds:
            try:
                recalculated_params = self.db.recalculate_aggregated_model(
                    round_num, exclude_withdrawn=True
                )
                if recalculated_params is not None:
                    self.db.save_aggregated_model(
                        round_num=round_num,
                        parameters=recalculated_params,
                        metrics={
                            "propagated_unlearning": True,
                            "method": "recalculation_after_hessian_unlearning",
                        },
                    )
                    print(f"[Hessian Unlearning] Propagated to round {round_num}")
            except Exception as e:
                print(f"[Hessian Unlearning] Failed to propagate to round {round_num}: {e}")

    def unlearn_samples_all_rounds(
        self,
        client_id: int,
        sample_ids: List[int],
        dataset_name: str = "MNIST",
        propagate: bool = True,
        influence_scale: float = 1.0,
    ) -> dict:
        """
        Unlearn specific samples from a client's contributions across all rounds using IHVP.

        Finds every round where ``sample_ids`` appear in the client's training
        trace and applies second-order per-sample unlearning to each one.  If no
        sample manifest is stored for a round, all target samples are assumed
        present (conservative fallback).

        Args:
            client_id: ID of the client that trained on the samples.
            sample_ids: Global sample IDs to remove.
            dataset_name: Dataset name (used for propagation metadata).
            propagate: If True, recalculate aggregated models for subsequent
                       rounds using the corrected per-sample weights.
            influence_scale: Additional multiplier on the IHVP correction.

        Returns:
            Dictionary with keys ``client_id``, ``sample_ids``,
            ``sample_rounds``, ``unlearned_rounds``, ``propagated``.
        """
        if not sample_ids:
            raise ValueError("sample_ids must be non-empty")

        sample_rounds = self.db.find_rounds_containing_samples(client_id, sample_ids)

        results = {
            "client_id": client_id,
            "sample_ids": sorted(int(s) for s in sample_ids),
            "sample_rounds": sample_rounds,
            "unlearned_rounds": [],
            "propagated": propagate,
            "method": "hessian_ihvp_per_sample",
        }

        if not sample_rounds:
            results["error"] = (
                f"Client {client_id} has no contributions containing the requested samples"
            )
            return results

        for round_num in sorted(sample_rounds.keys()):
            round_sample_ids = sample_rounds[round_num]
            try:
                unlearned_params, metadata = self.unlearn_samples_from_contribution(
                    round_num=round_num,
                    client_id=client_id,
                    sample_ids=round_sample_ids,
                    influence_scale=influence_scale,
                )
                self.db.save_aggregated_model(
                    round_num=round_num,
                    parameters=unlearned_params,
                    metrics={
                        "unlearned_samples": True,
                        "unlearned_client": client_id,
                        "unlearned_sample_ids": metadata["sample_ids"],
                        "method": "hessian_ihvp_per_sample",
                        "influence_scale": influence_scale,
                    },
                )
                results["unlearned_rounds"].append({
                    "round": round_num,
                    "metadata": metadata,
                    "status": "success",
                })
            except Exception as e:
                results["unlearned_rounds"].append({
                    "round": round_num,
                    "status": "failed",
                    "error": str(e),
                })

        if propagate and results["unlearned_rounds"]:
            self._propagate_sample_unlearning(
                from_round=max(sample_rounds.keys()),
                dataset_name=dataset_name,
            )

        return results

    def _propagate_sample_unlearning(self, from_round: int, dataset_name: str) -> List[int]:
        """
        Propagate per-sample unlearning to rounds STRICTLY AFTER ``from_round``.

        Recalculates aggregated models for subsequent rounds, which now account
        for per-sample withdrawals by reducing fed-averaging weights.

        Args:
            from_round: LATEST round where per-sample unlearning was applied
                (typically ``max(sample_rounds.keys())``). Pass the latest, not
                the earliest, so per-round aggregate corrections are preserved.
            dataset_name: Dataset name (for logging).

        Returns:
            List of round numbers that were recalculated.
        """
        stats = self.db.get_statistics()
        all_rounds = sorted(stats["rounds"].keys())
        subsequent_rounds = [r for r in all_rounds if r > from_round]

        if not subsequent_rounds:
            return []

        print(f"[Hessian Unlearning] Propagating sample unlearning to rounds: {subsequent_rounds}")
        updated = []

        for round_num in subsequent_rounds:
            try:
                recalculated_params = self.db.recalculate_aggregated_model(
                    round_num, exclude_withdrawn=True
                )
                if recalculated_params is not None:
                    self.db.save_aggregated_model(
                        round_num=round_num,
                        parameters=recalculated_params,
                        metrics={
                            "propagated_sample_unlearning": True,
                            "method": "recalculation_after_hessian_sample_unlearning",
                        },
                    )
                    updated.append(round_num)
                    print(f"[Hessian Unlearning] Propagated sample unlearning to round {round_num}")
            except Exception as e:
                print(f"[Hessian Unlearning] Failed to propagate sample unlearning to round {round_num}: {e}")

        return updated

    def _load_aggregated_model(self, round_num: int) -> Optional[List[np.ndarray]]:
        """Load the aggregated model for a specific round."""
        round_dir = self.db.contributions_dir / f"round_{round_num:04d}"
        if not round_dir.exists():
            return None
        aggregated_files = list(round_dir.glob("round_*_aggregated_*_params.pkl"))
        if not aggregated_files:
            return None
        latest_file = max(aggregated_files, key=lambda p: p.stat().st_mtime)
        with open(latest_file, "rb") as f:
            return pickle.load(f)


class ClassDiscriminativePruningUnlearning:
    """
    Class-discriminative pruning for federated unlearning (Wang et al., WWW 2022:
    *Federated Unlearning via Class-Discriminative Pruning*).

    Identifies CNN conv-channel filters that are highly discriminative for the
    target class(es) via TF-IDF analysis on per-class mean activations, then
    zeroes those filters' weights and biases in the global aggregate. Optional
    fine-tune step recovers utility on retain data.

    The paper targets CLASS-level unlearning. For sample-level unlearning, the
    target classes are the unique labels of the forget samples — so pruning
    affects ALL samples of those classes, not only the forget set. This
    limitation is intentional and surfaced in the benchmark.

    Args:
        contribution_db: ContributionDB instance for accessing contributions.
        model: A PyTorch model (its architecture is used; params are overwritten).
        dataset_name: Dataset name passed to ``load_data`` for probe + retain loaders.
        num_clients: Number of FL clients (matches the original training run).
        device: Torch device. Defaults to CUDA if available.
        prune_ratio: Fraction of channels to zero per conv layer per target class
                     (e.g. 0.1 → top 10% of channels by TF-IDF for that class).
        n_probe_per_class: Probe samples per class for TF-IDF computation.
        finetune_epochs: 0 → no fine-tune; >0 → SGD fine-tune on retain data.
        finetune_lr: Learning rate for the fine-tune step.
        batch_size: DataLoader batch size for probe + fine-tune.
    """

    def __init__(
        self,
        contribution_db: ContributionDB,
        model: torch.nn.Module,
        dataset_name: str,
        num_clients: int,
        device: Optional[torch.device] = None,
        prune_ratio: float = 0.1,
        n_probe_per_class: int = 64,
        finetune_epochs: int = 0,
        finetune_lr: float = 1e-3,
        batch_size: int = 32,
    ):
        self.db = contribution_db
        self.model = model
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prune_ratio = float(prune_ratio)
        self.n_probe_per_class = int(n_probe_per_class)
        self.finetune_epochs = int(finetune_epochs)
        self.finetune_lr = float(finetune_lr)
        self.batch_size = int(batch_size)
        self._client_dataset_cache: Dict[int, object] = {}

    # ------------------------------------------------------------------
    # Data + model helpers (mirror the HessianInfluenceUnlearning pattern)
    # ------------------------------------------------------------------

    def _base_dataset_name(self) -> str:
        return self.dataset_name.split("_")[0] if "_" in self.dataset_name else self.dataset_name

    def _get_client_loaders(self):
        """Reload client loaders (cached per call). Returns list of DataLoaders."""
        client_loaders, _ = load_data(
            self._base_dataset_name(),
            num_clients=self.num_clients,
            batch_size=self.batch_size,
        )
        return client_loaders

    def _get_client_dataset(self, client_id: int):
        if client_id in self._client_dataset_cache:
            return self._client_dataset_cache[client_id]
        client_loaders = self._get_client_loaders()
        ds = client_loaders[client_id].dataset
        self._client_dataset_cache[client_id] = ds
        return ds

    def _make_loader(self, dataset, sample_ids: Optional[List[int]] = None, shuffle: bool = False) -> DataLoader:
        if sample_ids is None:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        target = set(int(s) for s in sample_ids)
        indices = [i for i, gid in enumerate(dataset.global_ids) if int(gid) in target]
        if not indices:
            return None
        return DataLoader(Subset(dataset, indices), batch_size=self.batch_size, shuffle=shuffle)

    def _set_model_params(self, params: List[np.ndarray]):
        set_parameters(self.model, params)
        self.model.to(self.device)

    def _infer_num_classes(self) -> int:
        """Infer num_classes from the model's last Linear layer or dataset config."""
        last_linear = None
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                last_linear = m
        if last_linear is not None:
            return int(last_linear.out_features)
        cfg = DEFAULT_DATASET_CONFIGS.get(self._base_dataset_name().upper(), {})
        return int(cfg.get("num_classes", 10))

    # ------------------------------------------------------------------
    # TF-IDF computation + pruning logic
    # ------------------------------------------------------------------

    def _collect_target_classes(self, client_id: int, sample_ids: List[int]) -> List[int]:
        """Look up class labels for the given sample IDs in the client's dataset."""
        dataset = self._get_client_dataset(client_id)
        remaining = set(int(s) for s in sample_ids)
        classes: set = set()
        for i, gid in enumerate(dataset.global_ids):
            gid_int = int(gid)
            if gid_int in remaining:
                _, label, _ = dataset[i]
                classes.add(int(label))
                remaining.discard(gid_int)
                if not remaining:
                    break
        return sorted(classes)

    def _build_probe_loader(self, num_classes: int) -> DataLoader:
        """
        Build a balanced probe loader with up to ``n_probe_per_class`` samples per
        class, drawn from the union of all client partitions. Returns a DataLoader
        that yields (image, label) pairs.
        """
        client_loaders = self._get_client_loaders()
        per_class_buckets: Dict[int, List] = {k: [] for k in range(num_classes)}

        for cl in client_loaders:
            ds = cl.dataset  # WithSampleIds
            for i in range(len(ds)):
                _, label, _ = ds[i]
                k = int(label)
                if k in per_class_buckets and len(per_class_buckets[k]) < self.n_probe_per_class:
                    img, lbl, _ = ds[i]
                    per_class_buckets[k].append((img, int(lbl)))
            if all(len(b) >= self.n_probe_per_class for b in per_class_buckets.values()):
                break

        samples = []
        for k in sorted(per_class_buckets):
            samples.extend(per_class_buckets[k])

        class _ProbeDataset(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[idx]

        return DataLoader(_ProbeDataset(samples), batch_size=self.batch_size, shuffle=False)

    def _compute_tfidf(self, probe_loader: DataLoader, num_classes: int) -> Dict[str, torch.Tensor]:
        """
        Run the probe loader through the model with forward hooks on every Conv2d
        and return ``{layer_name: tfidf_matrix[num_classes, channels]}``.

        TF[ℓ, c, k] = mean spatial activation of channel c on samples of class k.
        IDF[ℓ, c]   = log(num_classes / (1 + |{k': mean_act[ℓ, c, k'] > τ}|)).
        TF-IDF      = TF * IDF.
        τ is the per-channel median across classes.
        """
        conv_layers = {
            name: m for name, m in self.model.named_modules() if isinstance(m, nn.Conv2d)
        }
        if not conv_layers:
            return {}

        sums: Dict[str, torch.Tensor] = {
            name: torch.zeros(num_classes, m.out_channels, device=self.device)
            for name, m in conv_layers.items()
        }
        counts = torch.zeros(num_classes, device=self.device)

        captured: Dict[str, torch.Tensor] = {}

        def make_hook(name):
            def hook(module, inputs, output):
                # output: [B, C, H, W] → spatial mean → [B, C]
                captured[name] = output.detach().mean(dim=[2, 3])
            return hook

        handles = [m.register_forward_hook(make_hook(name)) for name, m in conv_layers.items()]
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in probe_loader:
                    if len(batch) == 3:
                        images, labels, _ = batch
                    else:
                        images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    captured.clear()
                    self.model(images)
                    for k in range(num_classes):
                        mask = labels == k
                        if mask.any():
                            counts[k] += mask.sum()
                            for name, act in captured.items():
                                sums[name][k] += act[mask].sum(dim=0)
        finally:
            for h in handles:
                h.remove()

        counts_safe = counts.clamp_min(1.0).unsqueeze(1)  # [K, 1]
        tfidf_dict: Dict[str, torch.Tensor] = {}
        for name, total in sums.items():
            mean_act = total / counts_safe  # [K, C]
            # Per-channel median across classes (threshold τ)
            tau = mean_act.median(dim=0, keepdim=True).values  # [1, C]
            # Number of classes whose mean activation exceeds τ for each channel
            active_count = (mean_act > tau).sum(dim=0).float() + 1.0  # [C]
            idf = torch.log(torch.tensor(float(num_classes), device=self.device) / active_count)
            tfidf_dict[name] = mean_act * idf.unsqueeze(0)  # [K, C]
        return tfidf_dict

    def _apply_pruning(
        self,
        params: List[np.ndarray],
        target_classes: List[int],
        tfidf_dict: Dict[str, torch.Tensor],
    ) -> Tuple[List[np.ndarray], Dict[str, List[int]]]:
        """
        Zero out conv weight + bias slots for the top-``prune_ratio`` channels per
        target class, in each conv layer present in ``tfidf_dict``.
        Returns (pruned_params, channels_pruned_per_layer).
        """
        state_keys = list(self.model.state_dict().keys())
        new_params = [np.array(p, copy=True) for p in params]
        pruned_per_layer: Dict[str, List[int]] = {}

        for layer_name, tfidf in tfidf_dict.items():
            n_channels = tfidf.shape[1]
            if n_channels == 0 or self.prune_ratio <= 0.0:
                pruned_per_layer[layer_name] = []
                continue
            n_prune = max(1, int(round(self.prune_ratio * n_channels)))
            n_prune = min(n_prune, n_channels)
            prune_set: set = set()
            for k in target_classes:
                if 0 <= k < tfidf.shape[0]:
                    topk = torch.topk(tfidf[k], n_prune).indices.cpu().tolist()
                    prune_set.update(int(c) for c in topk)
            channels = sorted(prune_set)
            pruned_per_layer[layer_name] = channels

            weight_key = f"{layer_name}.weight"
            bias_key = f"{layer_name}.bias"
            if weight_key in state_keys:
                idx = state_keys.index(weight_key)
                for c in channels:
                    if c < new_params[idx].shape[0]:
                        new_params[idx][c] = 0.0
            if bias_key in state_keys:
                idx = state_keys.index(bias_key)
                for c in channels:
                    if c < new_params[idx].shape[0]:
                        new_params[idx][c] = 0.0

        return new_params, pruned_per_layer

    def _finetune_on_retain(
        self,
        params: List[np.ndarray],
        retain_loader: Optional[DataLoader],
    ) -> List[np.ndarray]:
        """Optional SGD fine-tune on retained data to recover utility lost from pruning."""
        if self.finetune_epochs <= 0 or retain_loader is None:
            return params
        self._set_model_params(params)
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.finetune_lr, momentum=0.9, weight_decay=1e-4
        )
        for _ in range(self.finetune_epochs):
            for batch in retain_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
        self.model.eval()
        return get_parameters(self.model)

    # ------------------------------------------------------------------
    # Public unlearning methods
    # ------------------------------------------------------------------

    def unlearn_samples_from_contribution(
        self,
        round_num: int,
        client_id: int,
        sample_ids: List[int],
        current_model_params: Optional[List[np.ndarray]] = None,
        influence_scale: float = 1.0,  # accepted for API parity; unused here
    ) -> Tuple[List[np.ndarray], dict]:
        if not sample_ids:
            raise ValueError("sample_ids must be non-empty")
        if current_model_params is None:
            current_model_params = self._load_aggregated_model(round_num)
            if current_model_params is None:
                raise ValueError(f"Could not load aggregated model for round {round_num}")

        self._set_model_params(current_model_params)
        target_classes = self._collect_target_classes(client_id, sample_ids)
        if not target_classes:
            raise ValueError(
                f"Could not determine target classes for client {client_id} samples {sample_ids}"
            )
        num_classes = self._infer_num_classes()
        probe_loader = self._build_probe_loader(num_classes)
        tfidf_dict = self._compute_tfidf(probe_loader, num_classes)
        pruned_params, pruned_channels = self._apply_pruning(
            current_model_params, target_classes, tfidf_dict
        )

        retain_loader = None
        if self.finetune_epochs > 0:
            client_dataset = self._get_client_dataset(client_id)
            forget_set = set(int(s) for s in sample_ids)
            retain_ids = [int(g) for g in client_dataset.global_ids if int(g) not in forget_set]
            retain_loader = (
                self._make_loader(client_dataset, retain_ids, shuffle=True) if retain_ids else None
            )
        pruned_params = self._finetune_on_retain(pruned_params, retain_loader)

        all_contributions = self.db.list_contributions(round_num=round_num, exclude_withdrawn=False)
        total_weight = sum(c["num_samples"] for c in all_contributions)
        client_weight = next(
            (c["num_samples"] for c in all_contributions if c["client_id"] == client_id), 0
        )

        metadata = {
            "round": round_num,
            "client_id": client_id,
            "sample_ids": sorted(int(s) for s in sample_ids),
            "target_classes": list(target_classes),
            "prune_ratio": self.prune_ratio,
            "n_probe_per_class": self.n_probe_per_class,
            "finetune_epochs": self.finetune_epochs,
            "pruned_channels_per_layer": {
                name: list(ch) for name, ch in pruned_channels.items()
            },
            "total_weight": total_weight,
            "client_weight": client_weight,
            "method": "class_discriminative_pruning",
        }
        return pruned_params, metadata

    def unlearn_client_contribution(
        self,
        round_num: int,
        client_id: int,
        current_model_params: Optional[List[np.ndarray]] = None,
        influence_scale: float = 1.0,
    ) -> Tuple[List[np.ndarray], dict]:
        if current_model_params is None:
            current_model_params = self._load_aggregated_model(round_num)
            if current_model_params is None:
                raise ValueError(f"Could not load aggregated model for round {round_num}")

        self._set_model_params(current_model_params)
        client_dataset = self._get_client_dataset(client_id)
        client_sample_ids = [int(g) for g in client_dataset.global_ids]
        # All client classes become targets
        target_classes = self._collect_target_classes(client_id, client_sample_ids)
        if not target_classes:
            raise ValueError(f"Client {client_id} has no samples (cannot determine target classes)")

        num_classes = self._infer_num_classes()
        probe_loader = self._build_probe_loader(num_classes)
        tfidf_dict = self._compute_tfidf(probe_loader, num_classes)
        pruned_params, pruned_channels = self._apply_pruning(
            current_model_params, target_classes, tfidf_dict
        )

        retain_loader = None
        if self.finetune_epochs > 0:
            # Retain = all OTHER clients' data
            other_loaders = [cl for i, cl in enumerate(self._get_client_loaders()) if i != client_id]
            retain_datasets = [cl.dataset for cl in other_loaders]
            if retain_datasets:
                combined = (
                    retain_datasets[0]
                    if len(retain_datasets) == 1
                    else ConcatDataset(retain_datasets)
                )
                retain_loader = DataLoader(combined, batch_size=self.batch_size, shuffle=True)
        pruned_params = self._finetune_on_retain(pruned_params, retain_loader)

        all_contributions = self.db.list_contributions(round_num=round_num, exclude_withdrawn=False)
        total_weight = sum(c["num_samples"] for c in all_contributions)
        client_weight = next(
            (c["num_samples"] for c in all_contributions if c["client_id"] == client_id), 0
        )

        metadata = {
            "round": round_num,
            "client_id": client_id,
            "target_classes": list(target_classes),
            "prune_ratio": self.prune_ratio,
            "n_probe_per_class": self.n_probe_per_class,
            "finetune_epochs": self.finetune_epochs,
            "pruned_channels_per_layer": {
                name: list(ch) for name, ch in pruned_channels.items()
            },
            "total_weight": total_weight,
            "client_weight": client_weight,
            "method": "class_discriminative_pruning",
        }
        return pruned_params, metadata

    def unlearn_client_all_rounds(
        self,
        client_id: int,
        dataset_name: str = "MNIST",
        propagate: bool = True,
        influence_scale: float = 1.0,
    ) -> dict:
        all_contributions = self.db.list_contributions(exclude_withdrawn=False)
        affected_rounds = sorted(
            set(c["round"] for c in all_contributions if c["client_id"] == client_id)
        )
        if not affected_rounds:
            return {"error": f"Client {client_id} has no contributions"}

        results = {
            "client_id": client_id,
            "affected_rounds": affected_rounds,
            "unlearned_rounds": [],
            "propagated": propagate,
            "method": "class_discriminative_pruning",
        }

        for round_num in affected_rounds:
            try:
                unlearned_params, metadata = self.unlearn_client_contribution(round_num, client_id)
                self.db.save_aggregated_model(
                    round_num=round_num,
                    parameters=unlearned_params,
                    metrics={
                        "unlearned": True,
                        "unlearned_client": client_id,
                        "method": "class_discriminative_pruning",
                        "prune_ratio": self.prune_ratio,
                    },
                )
                results["unlearned_rounds"].append(
                    {"round": round_num, "metadata": metadata, "status": "success"}
                )
            except Exception as e:
                results["unlearned_rounds"].append(
                    {"round": round_num, "status": "failed", "error": str(e)}
                )

        if propagate and results["unlearned_rounds"]:
            self._propagate_unlearning(affected_rounds, dataset_name)

        return results

    def unlearn_samples_all_rounds(
        self,
        client_id: int,
        sample_ids: List[int],
        dataset_name: str = "MNIST",
        propagate: bool = True,
        influence_scale: float = 1.0,
    ) -> dict:
        if not sample_ids:
            raise ValueError("sample_ids must be non-empty")

        sample_rounds = self.db.find_rounds_containing_samples(client_id, sample_ids)
        results = {
            "client_id": client_id,
            "sample_ids": sorted(int(s) for s in sample_ids),
            "sample_rounds": sample_rounds,
            "unlearned_rounds": [],
            "propagated": propagate,
            "method": "class_discriminative_pruning",
        }
        if not sample_rounds:
            results["error"] = (
                f"Client {client_id} has no contributions containing the requested samples"
            )
            return results

        for round_num in sorted(sample_rounds.keys()):
            round_sample_ids = sample_rounds[round_num]
            try:
                unlearned_params, metadata = self.unlearn_samples_from_contribution(
                    round_num=round_num, client_id=client_id, sample_ids=round_sample_ids
                )
                self.db.save_aggregated_model(
                    round_num=round_num,
                    parameters=unlearned_params,
                    metrics={
                        "unlearned_samples": True,
                        "unlearned_client": client_id,
                        "unlearned_sample_ids": metadata["sample_ids"],
                        "method": "class_discriminative_pruning",
                        "prune_ratio": self.prune_ratio,
                    },
                )
                results["unlearned_rounds"].append(
                    {"round": round_num, "metadata": metadata, "status": "success"}
                )
            except Exception as e:
                results["unlearned_rounds"].append(
                    {"round": round_num, "status": "failed", "error": str(e)}
                )

        if propagate and results["unlearned_rounds"]:
            self._propagate_sample_unlearning(max(sample_rounds.keys()), dataset_name)
        return results

    def _propagate_unlearning(self, affected_rounds: List[int], dataset_name: str):
        stats = self.db.get_statistics()
        all_rounds = sorted(stats["rounds"].keys())
        max_affected = max(affected_rounds)
        subsequent_rounds = [r for r in all_rounds if r > max_affected]
        if not subsequent_rounds:
            return
        print(f"[Class Pruning] Propagating to rounds: {subsequent_rounds}")
        for round_num in subsequent_rounds:
            try:
                recalculated = self.db.recalculate_aggregated_model(round_num, exclude_withdrawn=True)
                if recalculated is not None:
                    self.db.save_aggregated_model(
                        round_num=round_num,
                        parameters=recalculated,
                        metrics={
                            "propagated_unlearning": True,
                            "method": "recalculation_after_class_pruning",
                        },
                    )
                    print(f"[Class Pruning] Propagated to round {round_num}")
            except Exception as e:
                print(f"[Class Pruning] Failed to propagate to round {round_num}: {e}")

    def _propagate_sample_unlearning(self, from_round: int, dataset_name: str) -> List[int]:
        stats = self.db.get_statistics()
        all_rounds = sorted(stats["rounds"].keys())
        subsequent_rounds = [r for r in all_rounds if r > from_round]
        if not subsequent_rounds:
            return []
        print(f"[Class Pruning] Propagating sample unlearning to rounds: {subsequent_rounds}")
        updated = []
        for round_num in subsequent_rounds:
            try:
                recalculated = self.db.recalculate_aggregated_model(round_num, exclude_withdrawn=True)
                if recalculated is not None:
                    self.db.save_aggregated_model(
                        round_num=round_num,
                        parameters=recalculated,
                        metrics={
                            "propagated_sample_unlearning": True,
                            "method": "recalculation_after_class_pruning_sample",
                        },
                    )
                    updated.append(round_num)
            except Exception as e:
                print(f"[Class Pruning] Failed to propagate sample unlearning to round {round_num}: {e}")
        return updated

    def _load_aggregated_model(self, round_num: int) -> Optional[List[np.ndarray]]:
        round_dir = self.db.contributions_dir / f"round_{round_num:04d}"
        if not round_dir.exists():
            return None
        agg_files = list(round_dir.glob("round_*_aggregated_*_params.pkl"))
        if not agg_files:
            return None
        latest = max(agg_files, key=lambda p: p.stat().st_mtime)
        with open(latest, "rb") as f:
            return pickle.load(f)


class GradientAscentKDUnlearning:
    """
    Gradient ascent with knowledge distillation for federated unlearning
    (SCRUB-style; Kurmanji et al., NeurIPS 2023:
    *Towards Unbounded Machine Unlearning*).

    Maintains a frozen teacher = θ_agg (the pre-unlearning aggregate). Fine-tunes
    a trainable student (initialized to θ_agg) for ``epochs`` outer iterations:

      - **Forget step (max):** for each batch in the forget loader, compute
        L_forget = -β_forget · CE(student(x), y); backprop and step.
      - **Retain step (min):** for each batch in the retain loader, compute
        L_retain = α_retain · CE(student(x), y) + γ_kd · KD(student || teacher);
        backprop and step.

    KD = T² · KL(softmax(student/T), softmax(teacher/T)) (standard distillation).
    Returns the student parameters.

    Args:
        contribution_db: ContributionDB instance.
        model: PyTorch model (student/teacher are deepcopies of this).
        dataset_name: Dataset name passed to ``load_data``.
        num_clients: Number of FL clients (matches the original training run).
        device: Torch device. Defaults to CUDA if available.
        epochs: Outer epochs over the retain set.
        lr: AdamW learning rate.
        alpha_retain: Weight on retain CE loss.
        gamma_kd: Weight on KD loss.
        beta_forget: Weight on forget ascent loss.
        temperature: KD temperature.
        max_forget_steps: Cap forget batches per epoch (None → unlimited).
        batch_size: DataLoader batch size.
    """

    def __init__(
        self,
        contribution_db: ContributionDB,
        model: torch.nn.Module,
        dataset_name: str,
        num_clients: int,
        device: Optional[torch.device] = None,
        epochs: int = 3,
        lr: float = 1e-3,
        alpha_retain: float = 1.0,
        gamma_kd: float = 1.0,
        beta_forget: float = 1.0,
        temperature: float = 4.0,
        max_forget_steps: Optional[int] = None,
        batch_size: int = 32,
    ):
        self.db = contribution_db
        self.model = model
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.alpha_retain = float(alpha_retain)
        self.gamma_kd = float(gamma_kd)
        self.beta_forget = float(beta_forget)
        self.temperature = float(temperature)
        self.max_forget_steps = max_forget_steps
        self.batch_size = int(batch_size)
        self._client_loaders_cache = None

    # ------------------------------------------------------------------
    # Data + model helpers
    # ------------------------------------------------------------------

    def _base_dataset_name(self) -> str:
        return self.dataset_name.split("_")[0] if "_" in self.dataset_name else self.dataset_name

    def _get_client_loaders(self):
        if self._client_loaders_cache is None:
            client_loaders, _ = load_data(
                self._base_dataset_name(),
                num_clients=self.num_clients,
                batch_size=self.batch_size,
            )
            self._client_loaders_cache = client_loaders
        return self._client_loaders_cache

    def _get_client_dataset(self, client_id: int):
        return self._get_client_loaders()[client_id].dataset

    def _make_loader(
        self, dataset, sample_ids: Optional[List[int]] = None, shuffle: bool = False
    ) -> Optional[DataLoader]:
        if sample_ids is None:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        target = set(int(s) for s in sample_ids)
        indices = [i for i, gid in enumerate(dataset.global_ids) if int(gid) in target]
        if not indices:
            return None
        return DataLoader(Subset(dataset, indices), batch_size=self.batch_size, shuffle=shuffle)

    def _build_teacher(self, params: List[np.ndarray]) -> torch.nn.Module:
        teacher = copy.deepcopy(self.model)
        set_parameters(teacher, params)
        teacher.to(self.device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        return teacher

    def _build_student(self, params: List[np.ndarray]) -> torch.nn.Module:
        student = copy.deepcopy(self.model)
        set_parameters(student, params)
        student.to(self.device)
        student.train()
        for p in student.parameters():
            p.requires_grad_(True)
        return student

    def _build_retain_loader(
        self, client_id: int, forget_sample_ids: List[int]
    ) -> Optional[DataLoader]:
        """All clients' data minus the forget samples for ``client_id``."""
        forget_set = set(int(s) for s in forget_sample_ids)
        client_loaders = self._get_client_loaders()
        retain_parts: List = []
        for i, cl in enumerate(client_loaders):
            ds = cl.dataset
            if i == client_id and forget_set:
                retain_indices = [
                    j for j, gid in enumerate(ds.global_ids) if int(gid) not in forget_set
                ]
                if retain_indices:
                    retain_parts.append(Subset(ds, retain_indices))
            else:
                retain_parts.append(ds)
        if not retain_parts:
            return None
        combined = retain_parts[0] if len(retain_parts) == 1 else ConcatDataset(retain_parts)
        return DataLoader(combined, batch_size=self.batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # Loss + training-loop helpers
    # ------------------------------------------------------------------

    def _kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        return F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

    @staticmethod
    def _unpack(batch):
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        return images, labels

    def _forget_step(self, student, optimizer, batch):
        images, labels = self._unpack(batch)
        images, labels = images.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        logits = student(images)
        loss = -self.beta_forget * F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())

    def _retain_step(self, student, teacher, optimizer, batch):
        images, labels = self._unpack(batch)
        images, labels = images.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        s_logits = student(images)
        with torch.no_grad():
            t_logits = teacher(images)
        ce = F.cross_entropy(s_logits, labels)
        kd = self._kd_loss(s_logits, t_logits)
        loss = self.alpha_retain * ce + self.gamma_kd * kd
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())

    def _run_finetune(
        self,
        params: List[np.ndarray],
        forget_loader: Optional[DataLoader],
        retain_loader: Optional[DataLoader],
    ) -> Tuple[List[np.ndarray], dict]:
        """Run the alternating max/min loop for ``self.epochs`` epochs."""
        if self.epochs <= 0:
            return params, {"forget_steps": 0, "retain_steps": 0}
        student = self._build_student(params)
        teacher = self._build_teacher(params)
        optimizer = torch.optim.AdamW(student.parameters(), lr=self.lr)

        forget_steps = 0
        retain_steps = 0
        for _ in range(self.epochs):
            if forget_loader is not None:
                for i, batch in enumerate(forget_loader):
                    if self.max_forget_steps is not None and i >= self.max_forget_steps:
                        break
                    self._forget_step(student, optimizer, batch)
                    forget_steps += 1
            if retain_loader is not None:
                for batch in retain_loader:
                    self._retain_step(student, teacher, optimizer, batch)
                    retain_steps += 1
        student.eval()
        return get_parameters(student), {
            "forget_steps": forget_steps,
            "retain_steps": retain_steps,
        }

    # ------------------------------------------------------------------
    # Public unlearning methods
    # ------------------------------------------------------------------

    def unlearn_samples_from_contribution(
        self,
        round_num: int,
        client_id: int,
        sample_ids: List[int],
        current_model_params: Optional[List[np.ndarray]] = None,
        influence_scale: float = 1.0,  # accepted for API parity; unused here
    ) -> Tuple[List[np.ndarray], dict]:
        if not sample_ids:
            raise ValueError("sample_ids must be non-empty")
        if current_model_params is None:
            current_model_params = self._load_aggregated_model(round_num)
            if current_model_params is None:
                raise ValueError(f"Could not load aggregated model for round {round_num}")

        client_dataset = self._get_client_dataset(client_id)
        forget_loader = self._make_loader(client_dataset, sample_ids, shuffle=True)
        retain_loader = self._build_retain_loader(client_id, sample_ids)

        unlearned_params, step_counts = self._run_finetune(
            current_model_params, forget_loader, retain_loader
        )

        all_contributions = self.db.list_contributions(round_num=round_num, exclude_withdrawn=False)
        total_weight = sum(c["num_samples"] for c in all_contributions)
        client_weight = next(
            (c["num_samples"] for c in all_contributions if c["client_id"] == client_id), 0
        )

        metadata = {
            "round": round_num,
            "client_id": client_id,
            "sample_ids": sorted(int(s) for s in sample_ids),
            "epochs": self.epochs,
            "lr": self.lr,
            "alpha_retain": self.alpha_retain,
            "gamma_kd": self.gamma_kd,
            "beta_forget": self.beta_forget,
            "temperature": self.temperature,
            "max_forget_steps": self.max_forget_steps,
            "forget_steps_taken": step_counts["forget_steps"],
            "retain_steps_taken": step_counts["retain_steps"],
            "total_weight": total_weight,
            "client_weight": client_weight,
            "method": "gradient_ascent_kd_unlearning",
        }
        return unlearned_params, metadata

    def unlearn_client_contribution(
        self,
        round_num: int,
        client_id: int,
        current_model_params: Optional[List[np.ndarray]] = None,
        influence_scale: float = 1.0,
    ) -> Tuple[List[np.ndarray], dict]:
        if current_model_params is None:
            current_model_params = self._load_aggregated_model(round_num)
            if current_model_params is None:
                raise ValueError(f"Could not load aggregated model for round {round_num}")

        client_dataset = self._get_client_dataset(client_id)
        forget_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
        retain_loader = self._build_retain_loader(client_id, [int(g) for g in client_dataset.global_ids])

        unlearned_params, step_counts = self._run_finetune(
            current_model_params, forget_loader, retain_loader
        )

        all_contributions = self.db.list_contributions(round_num=round_num, exclude_withdrawn=False)
        total_weight = sum(c["num_samples"] for c in all_contributions)
        client_weight = next(
            (c["num_samples"] for c in all_contributions if c["client_id"] == client_id), 0
        )

        metadata = {
            "round": round_num,
            "client_id": client_id,
            "epochs": self.epochs,
            "lr": self.lr,
            "alpha_retain": self.alpha_retain,
            "gamma_kd": self.gamma_kd,
            "beta_forget": self.beta_forget,
            "temperature": self.temperature,
            "max_forget_steps": self.max_forget_steps,
            "forget_steps_taken": step_counts["forget_steps"],
            "retain_steps_taken": step_counts["retain_steps"],
            "total_weight": total_weight,
            "client_weight": client_weight,
            "method": "gradient_ascent_kd_unlearning",
        }
        return unlearned_params, metadata

    def unlearn_client_all_rounds(
        self,
        client_id: int,
        dataset_name: str = "MNIST",
        propagate: bool = True,
        influence_scale: float = 1.0,
    ) -> dict:
        all_contributions = self.db.list_contributions(exclude_withdrawn=False)
        affected_rounds = sorted(
            set(c["round"] for c in all_contributions if c["client_id"] == client_id)
        )
        if not affected_rounds:
            return {"error": f"Client {client_id} has no contributions"}

        results = {
            "client_id": client_id,
            "affected_rounds": affected_rounds,
            "unlearned_rounds": [],
            "propagated": propagate,
            "method": "gradient_ascent_kd",
        }

        for round_num in affected_rounds:
            try:
                unlearned_params, metadata = self.unlearn_client_contribution(round_num, client_id)
                self.db.save_aggregated_model(
                    round_num=round_num,
                    parameters=unlearned_params,
                    metrics={
                        "unlearned": True,
                        "unlearned_client": client_id,
                        "method": "gradient_ascent_kd_unlearning",
                    },
                )
                results["unlearned_rounds"].append(
                    {"round": round_num, "metadata": metadata, "status": "success"}
                )
            except Exception as e:
                results["unlearned_rounds"].append(
                    {"round": round_num, "status": "failed", "error": str(e)}
                )
        if propagate and results["unlearned_rounds"]:
            self._propagate_unlearning(affected_rounds, dataset_name)
        return results

    def unlearn_samples_all_rounds(
        self,
        client_id: int,
        sample_ids: List[int],
        dataset_name: str = "MNIST",
        propagate: bool = True,
        influence_scale: float = 1.0,
    ) -> dict:
        if not sample_ids:
            raise ValueError("sample_ids must be non-empty")
        sample_rounds = self.db.find_rounds_containing_samples(client_id, sample_ids)
        results = {
            "client_id": client_id,
            "sample_ids": sorted(int(s) for s in sample_ids),
            "sample_rounds": sample_rounds,
            "unlearned_rounds": [],
            "propagated": propagate,
            "method": "gradient_ascent_kd",
        }
        if not sample_rounds:
            results["error"] = (
                f"Client {client_id} has no contributions containing the requested samples"
            )
            return results

        for round_num in sorted(sample_rounds.keys()):
            round_sample_ids = sample_rounds[round_num]
            try:
                unlearned_params, metadata = self.unlearn_samples_from_contribution(
                    round_num=round_num, client_id=client_id, sample_ids=round_sample_ids
                )
                self.db.save_aggregated_model(
                    round_num=round_num,
                    parameters=unlearned_params,
                    metrics={
                        "unlearned_samples": True,
                        "unlearned_client": client_id,
                        "unlearned_sample_ids": metadata["sample_ids"],
                        "method": "gradient_ascent_kd_unlearning",
                    },
                )
                results["unlearned_rounds"].append(
                    {"round": round_num, "metadata": metadata, "status": "success"}
                )
            except Exception as e:
                results["unlearned_rounds"].append(
                    {"round": round_num, "status": "failed", "error": str(e)}
                )

        if propagate and results["unlearned_rounds"]:
            self._propagate_sample_unlearning(max(sample_rounds.keys()), dataset_name)
        return results

    def _propagate_unlearning(self, affected_rounds: List[int], dataset_name: str):
        stats = self.db.get_statistics()
        all_rounds = sorted(stats["rounds"].keys())
        max_affected = max(affected_rounds)
        subsequent_rounds = [r for r in all_rounds if r > max_affected]
        if not subsequent_rounds:
            return
        print(f"[GA+KD Unlearning] Propagating to rounds: {subsequent_rounds}")
        for round_num in subsequent_rounds:
            try:
                recalculated = self.db.recalculate_aggregated_model(round_num, exclude_withdrawn=True)
                if recalculated is not None:
                    self.db.save_aggregated_model(
                        round_num=round_num,
                        parameters=recalculated,
                        metrics={
                            "propagated_unlearning": True,
                            "method": "recalculation_after_gradient_ascent_kd",
                        },
                    )
            except Exception as e:
                print(f"[GA+KD Unlearning] Failed to propagate to round {round_num}: {e}")

    def _propagate_sample_unlearning(self, from_round: int, dataset_name: str) -> List[int]:
        stats = self.db.get_statistics()
        all_rounds = sorted(stats["rounds"].keys())
        subsequent_rounds = [r for r in all_rounds if r > from_round]
        if not subsequent_rounds:
            return []
        updated = []
        for round_num in subsequent_rounds:
            try:
                recalculated = self.db.recalculate_aggregated_model(round_num, exclude_withdrawn=True)
                if recalculated is not None:
                    self.db.save_aggregated_model(
                        round_num=round_num,
                        parameters=recalculated,
                        metrics={
                            "propagated_sample_unlearning": True,
                            "method": "recalculation_after_gradient_ascent_kd_sample",
                        },
                    )
                    updated.append(round_num)
            except Exception as e:
                print(
                    f"[GA+KD Unlearning] Failed to propagate sample unlearning to round {round_num}: {e}"
                )
        return updated

    def _load_aggregated_model(self, round_num: int) -> Optional[List[np.ndarray]]:
        round_dir = self.db.contributions_dir / f"round_{round_num:04d}"
        if not round_dir.exists():
            return None
        agg_files = list(round_dir.glob("round_*_aggregated_*_params.pkl"))
        if not agg_files:
            return None
        latest = max(agg_files, key=lambda p: p.stat().st_mtime)
        with open(latest, "rb") as f:
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

