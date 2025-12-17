"""Script to test unlearning algorithms on a preserved original model."""
import argparse
import pickle
from pathlib import Path
import torch
import numpy as np

from contributions_db import ContributionDB
from unlearning import GradientBasedUnlearning, InfluenceFunctionBasedUnlearning
from model import create_model, set_parameters, get_parameters
from utils import load_data, test


def list_available_datasets(base_dir="contributions"):
    """List all available dataset names in the contributions directory."""
    from pathlib import Path
    contrib_dir = Path(base_dir)
    if not contrib_dir.exists():
        return []
    
    datasets = []
    for item in contrib_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it has round directories
            rounds = [d for d in item.iterdir() if d.is_dir() and d.name.startswith('round_')]
            if rounds:
                # Count actual round numbers
                round_numbers = []
                for round_dir in rounds:
                    try:
                        round_num = int(round_dir.name.split('_')[1])
                        round_numbers.append(round_num)
                    except (ValueError, IndexError):
                        continue
                if round_numbers:
                    datasets.append((item.name, len(round_numbers)))
    
    # Return as list of names (sorted)
    return sorted([name for name, _ in datasets])


def save_original_model_snapshot(
    contribution_db: ContributionDB,
    dataset_name: str,
    model_name: str,
    num_classes: int = None,
    num_channels: int = None,
    img_size: int = None,
    snapshot_name: str = "original_before_unlearning"
):
    """
    Save a snapshot of the original model before any unlearning.
    
    This preserves the original model state so different unlearning algorithms
    can be tested on the same starting point.
    
    Args:
        contribution_db: ContributionDB instance
        dataset_name: Dataset name
        model_name: Model architecture name
        num_classes: Number of classes (for custom datasets)
        num_channels: Number of channels (for custom datasets)
        img_size: Image size (for custom datasets)
        snapshot_name: Name for the snapshot
    """
    # Get the latest round
    stats = contribution_db.get_statistics()
    
    # If statistics are empty, try to scan the directory for rounds
    if not stats["rounds"]:
        # Check if the contributions directory exists
        if not contribution_db.contributions_dir.exists():
            available_datasets = list_available_datasets(contribution_db.base_dir)
            if available_datasets:
                error_msg = (
                    f"Contributions directory does not exist: {contribution_db.contributions_dir}\n"
                    f"Available datasets: {', '.join(available_datasets)}\n"
                    f"Note: For class_vertical partitioning, the dataset name might be '{dataset_name}_CLASS_VERTICAL' "
                    f"or just 'CUSTOM' (check what was used during training)."
                )
            else:
                error_msg = (
                    f"Contributions directory does not exist: {contribution_db.contributions_dir}\n"
                    f"Make sure you've run federated learning training first."
                )
            raise ValueError(error_msg)
        
        # Scan directory for round directories
        round_dirs = [
            d for d in contribution_db.contributions_dir.iterdir()
            if d.is_dir() and d.name.startswith('round_')
        ]
        
        if not round_dirs:
            available_datasets = list_available_datasets(contribution_db.base_dir)
            if available_datasets:
                error_msg = (
                    f"No round directories found in {contribution_db.contributions_dir}\n"
                    f"Available datasets: {', '.join(available_datasets)}\n"
                    f"Note: For class_vertical partitioning, the dataset name might be '{dataset_name}_CLASS_VERTICAL' "
                    f"or just 'CUSTOM' (check what was used during training)."
                )
            else:
                error_msg = (
                    f"No round directories found in {contribution_db.contributions_dir}\n"
                    f"Make sure you've run federated learning training first."
                )
            raise ValueError(error_msg)
        
        # Extract round numbers from directory names
        round_numbers = []
        for round_dir in round_dirs:
            try:
                round_num = int(round_dir.name.split('_')[1])
                round_numbers.append(round_num)
            except (ValueError, IndexError):
                continue
        
        if not round_numbers:
            raise ValueError(f"Could not parse round numbers from directory names in {contribution_db.contributions_dir}")
        
        latest_round = max(round_numbers)
        print(f"[Snapshot] Found {len(round_numbers)} rounds by scanning directory (statistics.json may be missing)")
        print(f"[Snapshot] Using latest round: {latest_round}")
    else:
        latest_round = max(stats["rounds"].keys())
    
    # Load the original aggregated model for the latest round
    round_dir = contribution_db.contributions_dir / f"round_{latest_round:04d}"
    aggregated_files = list(round_dir.glob("round_*_aggregated_*_params.pkl"))
    
    if not aggregated_files:
        raise ValueError(f"No aggregated model found for round {latest_round}")
    
    # Get the original (non-unlearned) model
    original_files = [f for f in aggregated_files if "unlearned" not in str(f)]
    if not original_files:
        # If all are unlearned, get the most recent one
        latest_file = max(aggregated_files, key=lambda p: p.stat().st_mtime)
    else:
        latest_file = max(original_files, key=lambda p: p.stat().st_mtime)
    
    # Load original parameters
    with open(latest_file, 'rb') as f:
        original_params = pickle.load(f)
    
    # Create snapshot directory
    snapshot_dir = contribution_db.contributions_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    
    # Save snapshot
    snapshot_file = snapshot_dir / f"{snapshot_name}_round_{latest_round:04d}_params.pkl"
    with open(snapshot_file, 'wb') as f:
        pickle.dump(original_params, f)
    
    # Save metadata
    snapshot_metadata = {
        "snapshot_name": snapshot_name,
        "round": latest_round,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "num_classes": num_classes,
        "num_channels": num_channels,
        "img_size": img_size,
        "source_file": str(latest_file),
        "parameters_count": len(original_params)
    }
    
    metadata_file = snapshot_dir / f"{snapshot_name}_round_{latest_round:04d}_metadata.json"
    import json
    with open(metadata_file, 'w') as f:
        json.dump(snapshot_metadata, f, indent=2)
    
    print(f"[Snapshot] Saved original model snapshot: {snapshot_file}")
    print(f"  Round: {latest_round}")
    print(f"  Parameters: {len(original_params)} layers")
    
    return snapshot_file, snapshot_metadata


def load_original_model_snapshot(
    contribution_db: ContributionDB,
    snapshot_name: str = "original_before_unlearning"
):
    """Load a saved original model snapshot."""
    snapshot_dir = contribution_db.contributions_dir / "snapshots"
    
    # Find snapshot files
    snapshot_files = list(snapshot_dir.glob(f"{snapshot_name}_round_*_params.pkl"))
    if not snapshot_files:
        raise ValueError(f"No snapshot found with name '{snapshot_name}'")
    
    # Get the latest snapshot
    latest_snapshot = max(snapshot_files, key=lambda p: p.stat().st_mtime)
    
    # Load parameters
    with open(latest_snapshot, 'rb') as f:
        params = pickle.load(f)
    
    # Load metadata
    metadata_file = latest_snapshot.parent / latest_snapshot.name.replace("_params.pkl", "_metadata.json")
    import json
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    print(f"[Snapshot] Loaded original model snapshot: {latest_snapshot}")
    return params, metadata


def test_unlearning_algorithm(
    contribution_db: ContributionDB,
    client_id: int,
    algorithm: str,
    snapshot_name: str = "original_before_unlearning",
    propagate: bool = False,
    evaluate: bool = True,
    dataset_path: str = None,
    **algorithm_kwargs
):
    """
    Test an unlearning algorithm on the original model snapshot.
    
    This loads the original model, applies unlearning, and saves the result
    separately so the original is preserved.
    
    Args:
        contribution_db: ContributionDB instance
        client_id: Client ID to unlearn
        algorithm: Algorithm name ("gradient" or "influence")
        snapshot_name: Name of the snapshot to use
        propagate: Whether to propagate unlearning to subsequent rounds
        evaluate: Whether to evaluate the unlearned model
        **algorithm_kwargs: Additional algorithm-specific parameters
    """
    print(f"\n{'='*70}")
    print(f"TESTING {algorithm.upper()}-BASED UNLEARNING ON ORIGINAL MODEL")
    print(f"{'='*70}")
    print(f"Client ID: {client_id}")
    print(f"Algorithm: {algorithm}")
    print(f"Snapshot: {snapshot_name}")
    print()
    
    # Load original model snapshot
    original_params, snapshot_metadata = load_original_model_snapshot(
        contribution_db, snapshot_name
    )
    
    # Initialize unlearner
    algorithm_name = algorithm.lower()
    if algorithm_name == "gradient":
        unlearner = GradientBasedUnlearning(contribution_db)
    elif algorithm_name == "influence":
        damping = algorithm_kwargs.get("damping_factor", 0.01)
        influence_scale = algorithm_kwargs.get("influence_scale", 1.0)
        unlearner = InfluenceFunctionBasedUnlearning(
            contribution_db, 
            damping_factor=damping
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'gradient' or 'influence'")
    
    # Get all rounds where client contributed
    all_contributions = contribution_db.list_contributions(exclude_withdrawn=False)
    
    # Find which clients actually contributed
    contributing_clients = sorted(set(contrib["client_id"] for contrib in all_contributions))
    
    affected_rounds = sorted(set(
        contrib["round"] for contrib in all_contributions 
        if contrib["client_id"] == client_id
    ))
    
    if not affected_rounds:
        error_msg = f"Client {client_id} has no contributions.\n"
        if contributing_clients:
            error_msg += f"Available clients with contributions: {contributing_clients}\n"
            error_msg += f"Please use one of these client IDs."
        else:
            error_msg += f"No contributions found in the database at all."
        raise ValueError(error_msg)
    
    print(f"Affected rounds: {affected_rounds}")
    print()
    
    # Unlearn in each round using the original model
    results = {
        "client_id": client_id,
        "algorithm": algorithm_name,
        "snapshot_name": snapshot_name,
        "affected_rounds": affected_rounds,
        "unlearned_rounds": []
    }
    
    for round_num in affected_rounds:
        print(f"Unlearning round {round_num}...")
        
        try:
            # Load the original aggregated model for this specific round
            round_dir = contribution_db.contributions_dir / f"round_{round_num:04d}"
            original_round_files = [
                f for f in round_dir.glob("round_*_aggregated_*_params.pkl")
                if "unlearned" not in str(f)
            ]
            
            if not original_round_files:
                raise ValueError(f"No original aggregated model found for round {round_num}")
            
            # Get the most recent original model for this round
            original_round_file = max(original_round_files, key=lambda p: p.stat().st_mtime)
            with open(original_round_file, 'rb') as f:
                original_round_params = pickle.load(f)
            
            if algorithm_name == "gradient":
                unlearned_params, metadata = unlearner.unlearn_client_contribution(
                    round_num=round_num,
                    client_id=client_id,
                    current_model_params=original_round_params
                )
            else:  # influence
                influence_scale = algorithm_kwargs.get("influence_scale", 1.0)
                unlearned_params, metadata = unlearner.unlearn_client_contribution(
                    round_num=round_num,
                    client_id=client_id,
                    current_model_params=original_round_params,
                    influence_scale=influence_scale
                )
            
            # Save unlearned model with algorithm identifier
            unlearned_suffix = f"unlearned_{algorithm_name}_{snapshot_name}"
            contribution_db.save_aggregated_model(
                round_num=round_num,
                parameters=unlearned_params,
                metrics={
                    "unlearned": True,
                    "unlearned_client": client_id,
                    "algorithm": algorithm_name,
                    "snapshot_name": snapshot_name,
                    "original_preserved": True,
                    **metadata
                },
                suffix=unlearned_suffix
            )
            
            results["unlearned_rounds"].append({
                "round": round_num,
                "metadata": metadata,
                "status": "success"
            })
            
            print(f"  ✓ Round {round_num} unlearned successfully")
            
        except Exception as e:
            results["unlearned_rounds"].append({
                "round": round_num,
                "status": "failed",
                "error": str(e)
            })
            print(f"  ✗ Round {round_num} failed: {e}")
    
    # Evaluate if requested
    if evaluate and results["unlearned_rounds"]:
        print(f"\nEvaluating unlearned models...")
        dataset_name = snapshot_metadata.get("dataset_name", "CUSTOM")
        model_name = snapshot_metadata.get("model_name", "resnet18")
        num_classes = snapshot_metadata.get("num_classes")
        num_channels = snapshot_metadata.get("num_channels")
        img_size = snapshot_metadata.get("img_size")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        net = create_model(
            model_name,
            dataset_name,
            num_classes=num_classes,
            num_channels=num_channels,
            img_size=img_size
        ).to(device)
        
        # Normalize dataset name for load_data (it only recognizes "CUSTOM", not "CUSTOM_CLASS_VERTICAL")
        load_dataset_name = dataset_name
        if "CUSTOM" in dataset_name.upper() or dataset_name.endswith("_CLASS_VERTICAL"):
            load_dataset_name = "CUSTOM"
        
        # Load test data
        _, testloader = load_data(
            load_dataset_name,
            num_clients=1,
            batch_size=32,
            dataset_path=dataset_path,
            img_size=img_size,
            num_channels=num_channels
        )
        
        # Evaluate each unlearned round
        for round_result in results["unlearned_rounds"]:
            if round_result["status"] != "success":
                continue
            
            round_num = round_result["round"]
            
            # Load unlearned model
            round_dir = contribution_db.contributions_dir / f"round_{round_num:04d}"
            unlearned_files = [
                f for f in round_dir.glob("round_*_aggregated_*_params.pkl")
                if f"unlearned_{algorithm_name}" in str(f) and snapshot_name in str(f)
            ]
            
            if unlearned_files:
                latest_file = max(unlearned_files, key=lambda p: p.stat().st_mtime)
                with open(latest_file, 'rb') as f:
                    unlearned_params = pickle.load(f)
                
                set_parameters(net, unlearned_params)
                loss, accuracy = test(net, testloader, device)
                
                print(f"  Round {round_num}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
                round_result["evaluation"] = {
                    "loss": float(loss),
                    "accuracy": float(accuracy)
                }
    
    print(f"\n{'='*70}")
    print(f"Unlearning test complete!")
    print(f"Original model preserved in snapshot: {snapshot_name}")
    print(f"Unlearned models saved with algorithm identifier: {algorithm_name}")
    print(f"{'='*70}\n")
    
    return results


def compare_algorithms_on_original(
    contribution_db: ContributionDB,
    client_id: int,
    snapshot_name: str = "original_before_unlearning",
    evaluate: bool = True,
    dataset_path: str = None
):
    """
    Compare both unlearning algorithms on the same original model.
    
    This ensures both algorithms start from the exact same model state.
    """
    print(f"\n{'='*70}")
    print(f"COMPARING UNLEARNING ALGORITHMS ON ORIGINAL MODEL")
    print(f"{'='*70}")
    print(f"Client ID: {client_id}")
    print(f"Snapshot: {snapshot_name}")
    print()
    
    # Test gradient-based
    print("1. Testing Gradient-Based Unlearning...")
    gradient_results = test_unlearning_algorithm(
        contribution_db,
        client_id,
        "gradient",
        snapshot_name=snapshot_name,
        propagate=False,
        evaluate=evaluate,
        dataset_path=dataset_path
    )
    
    # Test influence-based
    print("\n2. Testing Influence-Based Unlearning...")
    influence_results = test_unlearning_algorithm(
        contribution_db,
        client_id,
        "influence",
        snapshot_name=snapshot_name,
        propagate=False,
        evaluate=evaluate,
        dataset_path=dataset_path,
        damping_factor=0.01,
        influence_scale=1.0
    )
    
    # Compare results
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    if evaluate:
        print(f"\nEvaluation Results:")
        for round_num in gradient_results["affected_rounds"]:
            grad_result = next(
                (r for r in gradient_results["unlearned_rounds"] if r["round"] == round_num),
                None
            )
            inf_result = next(
                (r for r in influence_results["unlearned_rounds"] if r["round"] == round_num),
                None
            )
            
            if grad_result and inf_result and "evaluation" in grad_result and "evaluation" in inf_result:
                print(f"\n  Round {round_num}:")
                print(f"    Gradient-based:  Loss={grad_result['evaluation']['loss']:.4f}, Acc={grad_result['evaluation']['accuracy']:.4f}")
                print(f"    Influence-based: Loss={inf_result['evaluation']['loss']:.4f}, Acc={inf_result['evaluation']['accuracy']:.4f}")
    
    return {
        "gradient": gradient_results,
        "influence": influence_results
    }


def main():
    parser = argparse.ArgumentParser(description="Test unlearning algorithms on preserved original model")
    parser.add_argument("action", choices=["snapshot", "unlearn", "compare", "list-datasets", "list-clients"], 
                       help="Action to perform")
    parser.add_argument("--dataset", type=str,
                       help="Dataset name (required for snapshot/unlearn/compare)")
    parser.add_argument("--client-id", type=int,
                       help="Client ID to unlearn (required for unlearn/compare)")
    parser.add_argument("--algorithm", type=str, choices=["gradient", "influence"],
                       help="Unlearning algorithm (required for unlearn)")
    parser.add_argument("--snapshot-name", type=str, default="original_before_unlearning",
                       help="Name for the snapshot")
    parser.add_argument("--model", type=str, default="resnet18",
                       help="Model architecture name")
    parser.add_argument("--num-classes", type=int,
                       help="Number of classes (for custom datasets)")
    parser.add_argument("--num-channels", type=int,
                       help="Number of channels (for custom datasets)")
    parser.add_argument("--img-size", type=int,
                       help="Image size (for custom datasets)")
    parser.add_argument("--dataset-path", type=str,
                       help="Path to custom dataset (required for CUSTOM datasets)")
    parser.add_argument("--propagate", action="store_true",
                       help="Propagate unlearning to subsequent rounds")
    parser.add_argument("--no-evaluate", action="store_true",
                       help="Skip evaluation")
    
    args = parser.parse_args()
    
    # Handle list-datasets action
    if args.action == "list-datasets":
        print("Available datasets in contributions directory:")
        datasets = list_available_datasets()
        if datasets:
            for ds in datasets:
                # Count rounds by scanning directory
                from pathlib import Path
                base_dir = Path("contributions")
                ds_path = base_dir / ds
                if ds_path.exists():
                    round_dirs = [d for d in ds_path.iterdir() if d.is_dir() and d.name.startswith('round_')]
                    round_numbers = []
                    for round_dir in round_dirs:
                        try:
                            round_num = int(round_dir.name.split('_')[1])
                            round_numbers.append(round_num)
                        except (ValueError, IndexError):
                            continue
                    num_rounds = len(round_numbers)
                    if num_rounds > 0:
                        print(f"  - {ds} ({num_rounds} rounds, range: {min(round_numbers)}-{max(round_numbers)})")
        else:
            print("  No datasets found. Make sure you've run federated learning training first.")
        return
    
    # Handle list-clients action
    if args.action == "list-clients":
        if args.dataset is None:
            parser.error("--dataset is required for list-clients action")
        db = ContributionDB(dataset_name=args.dataset)
        
        # First, check directory structure
        print(f"\nChecking contributions directory: {db.contributions_dir}")
        print(f"Directory exists: {db.contributions_dir.exists()}")
        
        if not db.contributions_dir.exists():
            print(f"  ERROR: Directory does not exist!")
            return
        
        # Check for round directories
        round_dirs = [d for d in db.contributions_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('round_')]
        print(f"Found {len(round_dirs)} round directories")
        
        if round_dirs:
            # Check first round for client directories
            first_round = sorted(round_dirs)[0]
            print(f"\nInspecting first round: {first_round.name}")
            client_dirs = [d for d in first_round.iterdir() 
                          if d.is_dir() and d.name.startswith('client_')]
            print(f"  Found {len(client_dirs)} client directories")
            
            if client_dirs:
                # Check for metadata files
                first_client = client_dirs[0]
                print(f"  Inspecting first client: {first_client.name}")
                metadata_files = list(first_client.glob("*_metadata.json"))
                print(f"    Found {len(metadata_files)} metadata files")
                
                if metadata_files:
                    import json
                    with open(metadata_files[0], 'r') as f:
                        sample_meta = json.load(f)
                    print(f"    Sample metadata keys: {list(sample_meta.keys())}")
                    print(f"    Client ID in metadata: {sample_meta.get('client_id')}")
                    print(f"    Round in metadata: {sample_meta.get('round')}")
        
        # Now try to list contributions
        print(f"\nAttempting to load contributions...")
        all_contributions = db.list_contributions(exclude_withdrawn=False)
        print(f"Loaded {len(all_contributions)} contributions")
        
        contributing_clients = sorted(set(contrib["client_id"] for contrib in all_contributions))
        
        print(f"\nAvailable clients for dataset '{args.dataset}':")
        if contributing_clients:
            for client_id in contributing_clients:
                client_rounds = sorted(set(
                    contrib["round"] for contrib in all_contributions 
                    if contrib["client_id"] == client_id
                ))
                print(f"  Client {client_id}: {len(client_rounds)} rounds ({min(client_rounds)}-{max(client_rounds)})")
        else:
            print("  No clients found.")
            print("\n  Troubleshooting:")
            print("    - Check if round directories exist (round_0000, round_0001, etc.)")
            print("    - Check if client directories exist (client_0000, client_0001, etc.)")
            print("    - Check if metadata files exist (*_metadata.json)")
            print("    - Verify the dataset name matches what was used during training")
        return
    
    # Require dataset for other actions
    if args.dataset is None:
        parser.error("--dataset is required for this action")
    
    # Initialize contribution DB
    db = ContributionDB(dataset_name=args.dataset)
    
    if args.action == "snapshot":
        # Save original model snapshot
        save_original_model_snapshot(
            db,
            args.dataset,
            args.model,
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            img_size=args.img_size,
            snapshot_name=args.snapshot_name
        )
        
    elif args.action == "unlearn":
        if args.client_id is None:
            parser.error("--client-id is required for unlearn action")
        if args.algorithm is None:
            parser.error("--algorithm is required for unlearn action")
        
        # First, mark client as withdrawn if not already
        if not db.is_client_withdrawn(args.client_id):
            print(f"Marking client {args.client_id} as withdrawn...")
            db.withdraw_client(args.client_id, reason="Testing unlearning algorithms")
        
        # Test unlearning algorithm
        test_unlearning_algorithm(
            db,
            args.client_id,
            args.algorithm,
            snapshot_name=args.snapshot_name,
            propagate=args.propagate,
            evaluate=not args.no_evaluate,
            dataset_path=args.dataset_path
        )
        
    elif args.action == "compare":
        if args.client_id is None:
            parser.error("--client-id is required for compare action")
        
        # First, mark client as withdrawn if not already
        if not db.is_client_withdrawn(args.client_id):
            print(f"Marking client {args.client_id} as withdrawn...")
            db.withdraw_client(args.client_id, reason="Comparing unlearning algorithms")
        
        # Compare both algorithms
        compare_algorithms_on_original(
            db,
            args.client_id,
            snapshot_name=args.snapshot_name,
            evaluate=not args.no_evaluate,
            dataset_path=args.dataset_path
        )


if __name__ == "__main__":
    main()

