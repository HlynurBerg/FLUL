"""Tool to manage client withdrawals and recalculate models (git-like history management)."""
import argparse
from contributions_db import ContributionDB
from model import create_model, set_parameters, get_parameters
from utils import load_data, test
from unlearning import (
    GradientBasedUnlearning, 
    InfluenceFunctionBasedUnlearning,
    evaluate_unlearned_model
)
import torch


def withdraw_client(db, client_id, reason=None):
    """Withdraw a client from the federated learning system."""
    print(f"\n{'='*60}")
    print(f"WITHDRAWING CLIENT {client_id}")
    print(f"{'='*60}")
    
    withdrawal = db.withdraw_client(client_id, reason)
    print(f"\nWithdrawal Details:")
    print(f"  Client ID: {client_id}")
    print(f"  Withdrawn At: {withdrawal['withdrawn_at']}")
    print(f"  Reason: {withdrawal['reason'] or 'Not specified'}")
    print(f"  Affected Rounds: {withdrawal['affects_rounds']}")
    print(f"\nClient contributions are preserved but marked as withdrawn.")
    print(f"Use 'recalculate' command to rebuild models excluding this client.")
    print(f"{'='*60}\n")


def restore_client(db, client_id):
    """Restore a withdrawn client."""
    print(f"\n{'='*60}")
    print(f"RESTORING CLIENT {client_id}")
    print(f"{'='*60}")
    
    if db.restore_client(client_id):
        print(f"Client {client_id} has been restored.")
        print(f"Contributions are now active again.")
    else:
        print(f"Client {client_id} was not withdrawn.")
    print(f"{'='*60}\n")


def show_history(db, client_id=None, round_num=None):
    """Show contribution history (git log style)."""
    print(f"\n{'='*60}")
    print("CONTRIBUTION HISTORY")
    print(f"{'='*60}")
    
    history = db.get_history(client_id=client_id, round_num=round_num)
    
    if not history:
        print("No contributions found.")
        print(f"{'='*60}\n")
        return
    
    # Group by round
    current_round = None
    for entry in history:
        if entry["round"] != current_round:
            current_round = entry["round"]
            print(f"\nRound {entry['round']:04d}:")
            print("-" * 60)
        
        status = "WITHDRAWN" if entry["withdrawn"] else "ACTIVE"
        print(f"  [{status}] Client {entry['client_id']:04d} | "
              f"{entry['num_samples']} samples | {entry['timestamp']}")
        
        if entry.get("metrics"):
            metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                   for k, v in entry["metrics"].items()])
            print(f"    Metrics: {metrics_str}")
    
    print(f"\n{'='*60}\n")


def list_withdrawals(db):
    """List all withdrawn clients."""
    print(f"\n{'='*60}")
    print("WITHDRAWN CLIENTS")
    print(f"{'='*60}")
    
    withdrawn = db.get_withdrawn_clients()
    
    if not withdrawn:
        print("No clients are currently withdrawn.")
    else:
        for client_id in withdrawn:
            info = db.withdrawals["withdrawn_clients"][client_id]
            print(f"\nClient {client_id}:")
            print(f"  Withdrawn At: {info['withdrawn_at']}")
            print(f"  Reason: {info.get('reason', 'Not specified')}")
            print(f"  Affects Rounds: {info.get('affects_rounds', [])}")
    
    print(f"\n{'='*60}\n")


def recalculate_models(db, dataset_name, from_round=None, to_round=None, 
                      model_name="simplenet", num_classes=None, num_channels=None, 
                      img_size=None, dataset_path=None):
    """Recalculate aggregated models excluding withdrawn clients."""
    print(f"\n{'='*60}")
    print("RECALCULATING MODELS (EXCLUDING WITHDRAWN CLIENTS)")
    print(f"{'='*60}")
    
    # Get all rounds - scan directory if statistics are empty
    stats = db.get_statistics()
    if not stats.get("rounds"):
        # Scan directory for rounds
        round_dirs = [d for d in db.contributions_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('round_')]
        all_rounds = []
        for round_dir in round_dirs:
            try:
                round_num = int(round_dir.name.split('_')[1])
                all_rounds.append(round_num)
            except (ValueError, IndexError):
                continue
        all_rounds = sorted(all_rounds)
    else:
        all_rounds = sorted(stats["rounds"].keys())
    
    if from_round is not None:
        all_rounds = [r for r in all_rounds if r >= from_round]
    if to_round is not None:
        all_rounds = [r for r in all_rounds if r <= to_round]
    
    if not all_rounds:
        print("No rounds to recalculate.")
        print(f"{'='*60}\n")
        return
    
    print(f"Recalculating rounds: {all_rounds}")
    print()
    
    # Load model for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle dataset name (strip _CLASS_VERTICAL suffix if present)
    base_dataset = dataset_name.split("_")[0] if "_" in dataset_name else dataset_name
    
    if base_dataset == "CUSTOM":
        if num_classes is None or num_channels is None or img_size is None:
            raise ValueError("For CUSTOM datasets, --num-classes, --num-channels, and --img-size are required")
        net = create_model(model_name, base_dataset, 
                          num_classes=num_classes, num_channels=num_channels, img_size=img_size).to(device)
    else:
        net = create_model(model_name, base_dataset).to(device)
    
    _, testloader = load_data(
        base_dataset, 
        num_clients=1, 
        batch_size=32,
        dataset_path=dataset_path,
        img_size=img_size,
        num_channels=num_channels
    )
    
    for round_num in all_rounds:
        print(f"Recalculating Round {round_num:04d}...")
        
        # Recalculate aggregated model
        aggregated_params = db.recalculate_aggregated_model(round_num, exclude_withdrawn=True)
        
        if aggregated_params is None:
            print(f"  Failed: Insufficient contributions")
            continue
        
        # Evaluate the recalculated model
        set_parameters(net, aggregated_params)
        loss, accuracy = test(net, testloader, device)
        
        # Save the recalculated model
        db.save_aggregated_model(
            round_num=round_num,
            parameters=aggregated_params,
            metrics={"loss": float(loss), "accuracy": float(accuracy), "recalculated": True}
        )
        
        print(f"  Success: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
    
    print(f"\n{'='*60}\n")


def unlearn_client(db, dataset_name, client_id, algorithm="gradient", propagate=True, evaluate=True,
                   model_name="simplenet", num_classes=None, num_channels=None, img_size=None, 
                   dataset_path=None, **kwargs):
    """
    Unlearn a client's contributions using the specified unlearning algorithm.
    
    Args:
        db: ContributionDB instance
        dataset_name: Dataset name
        client_id: Client ID to unlearn
        algorithm: Algorithm to use ("gradient" or "influence")
        propagate: Whether to propagate to subsequent rounds
        evaluate: Whether to evaluate unlearned models
        **kwargs: Additional algorithm-specific parameters
    """
    algorithm_name = algorithm.lower()
    
    print(f"\n{'='*60}")
    if algorithm_name == "gradient":
        print(f"UNLEARNING CLIENT {client_id} (Gradient-Based Algorithm)")
        unlearner = GradientBasedUnlearning(db)
    elif algorithm_name == "influence":
        print(f"UNLEARNING CLIENT {client_id} (Influence Function-Based Algorithm)")
        damping = kwargs.get("damping_factor", 0.01)
        influence_scale = kwargs.get("influence_scale", 1.0)
        unlearner = InfluenceFunctionBasedUnlearning(db, damping_factor=damping)
    else:
        print(f"Error: Unknown algorithm '{algorithm}'. Use 'gradient' or 'influence'")
        return None
    
    print(f"{'='*60}")
    
    # Check if client is withdrawn
    if not db.is_client_withdrawn(client_id):
        print(f"Warning: Client {client_id} is not marked as withdrawn.")
        print(f"Proceeding with unlearning anyway...")
        print()
    
    # Perform unlearning
    print(f"\nApplying {algorithm_name}-based unlearning algorithm...")
    print(f"This will remove client {client_id}'s contributions without retraining from scratch.")
    print()
    
    if algorithm_name == "influence":
        results = unlearner.unlearn_client_all_rounds(
            client_id=client_id,
            dataset_name=dataset_name,
            propagate=propagate,
            influence_scale=influence_scale
        )
    else:
        results = unlearner.unlearn_client_all_rounds(
            client_id=client_id,
            dataset_name=dataset_name,
            propagate=propagate
        )
    
    if "error" in results:
        print(f"Error: {results['error']}")
        print(f"{'='*60}\n")
        return results
    
    print(f"\nUnlearning Results:")
    print(f"  Client ID: {client_id}")
    print(f"  Algorithm: {algorithm_name}")
    print(f"  Affected Rounds: {results['affected_rounds']}")
    print(f"  Propagate to subsequent rounds: {propagate}")
    print()
    
    # Evaluate unlearned models if requested
    if evaluate:
        print("Evaluating unlearned models...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Handle dataset name (strip _CLASS_VERTICAL suffix if present)
        base_dataset = dataset_name.split("_")[0] if "_" in dataset_name else dataset_name
        
        if base_dataset == "CUSTOM":
            if num_classes is None or num_channels is None or img_size is None:
                print("  Warning: Skipping evaluation - missing dataset parameters for CUSTOM dataset")
                evaluate = False
            else:
                net = create_model(model_name, base_dataset,
                                  num_classes=num_classes, num_channels=num_channels, img_size=img_size).to(device)
                _, testloader = load_data(
                    base_dataset,
                    num_clients=1,
                    batch_size=32,
                    dataset_path=dataset_path,
                    img_size=img_size,
                    num_channels=num_channels
                )
        else:
            net = create_model(model_name, base_dataset).to(device)
            _, testloader = load_data(base_dataset, num_clients=1, batch_size=32)
        
        for round_result in results["unlearned_rounds"]:
            if round_result["status"] == "success":
                round_num = round_result["round"]
                # Load unlearned model
                round_dir = db.contributions_dir / f"round_{round_num:04d}"
                aggregated_files = list(round_dir.glob("round_*_aggregated_*_params.pkl"))
                if aggregated_files:
                    # Get the most recent (should be the unlearned one)
                    latest_file = max(aggregated_files, key=lambda p: p.stat().st_mtime)
                    import pickle
                    with open(latest_file, 'rb') as f:
                        unlearned_params = pickle.load(f)
                    
                    # Evaluate
                    set_parameters(net, unlearned_params)
                    loss, accuracy = test(net, testloader, device)
                    
                    print(f"  Round {round_num:04d}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
                else:
                    print(f"  Round {round_num:04d}: Could not load model for evaluation")
            else:
                print(f"  Round {round_result['round']:04d}: {round_result.get('error', 'Failed')}")
    
    print(f"\nUnlearning complete!")
    print(f"Note: The unlearned models have been saved. Use 'recalculate' for full recalculation.")
    print(f"{'='*60}\n")
    
    return results


def compare_unlearning_algorithms(db, dataset_name, client_id, propagate=False, evaluate=True,
                                  model_name="simplenet", num_classes=None, num_channels=None,
                                  img_size=None, dataset_path=None):
    """
    Compare both unlearning algorithms on the same client.
    This helps analyze the differences between approaches.
    """
    print(f"\n{'='*60}")
    print(f"COMPARING UNLEARNING ALGORITHMS FOR CLIENT {client_id}")
    print(f"{'='*60}\n")
    
    import pickle
    import numpy as np
    
    # Get affected rounds
    all_contributions = db.list_contributions(exclude_withdrawn=False)
    affected_rounds = sorted(set(
        contrib["round"] for contrib in all_contributions 
        if contrib["client_id"] == client_id
    ))
    
    if not affected_rounds:
        print(f"Client {client_id} has no contributions.")
        return
    
    print(f"Affected rounds: {affected_rounds}\n")
    
    # Initialize both algorithms
    gradient_unlearner = GradientBasedUnlearning(db)
    influence_unlearner = InfluenceFunctionBasedUnlearning(db, damping_factor=0.01)
    
    # Load original models for comparison
    original_models = {}
    for round_num in affected_rounds:
        round_dir = db.contributions_dir / f"round_{round_num:04d}"
        aggregated_files = list(round_dir.glob("round_*_aggregated_*_params.pkl"))
        if aggregated_files:
            # Get the original (before unlearning)
            original_files = [f for f in aggregated_files if "unlearned" not in str(f)]
            if original_files:
                latest_original = max(original_files, key=lambda p: p.stat().st_mtime)
                with open(latest_original, 'rb') as f:
                    original_models[round_num] = pickle.load(f)
    
    # Run gradient-based unlearning
    print("=" * 60)
    print("1. GRADIENT-BASED UNLEARNING")
    print("=" * 60)
    gradient_results = gradient_unlearner.unlearn_client_all_rounds(
        client_id=client_id,
        dataset_name=dataset_name,
        propagate=propagate
    )
    
    # Load gradient-based models immediately after unlearning
    gradient_models = {}
    for round_result in gradient_results.get("unlearned_rounds", []):
        if round_result["status"] == "success":
            round_num = round_result["round"]
            # Load directly using the unlearner's method
            gradient_params = gradient_unlearner._load_aggregated_model(round_num)
            if gradient_params:
                gradient_models[round_num] = gradient_params
    
    # Run influence-based unlearning
    print("\n" + "=" * 60)
    print("2. INFLUENCE FUNCTION-BASED UNLEARNING")
    print("=" * 60)
    influence_results = influence_unlearner.unlearn_client_all_rounds(
        client_id=client_id,
        dataset_name=dataset_name,
        propagate=propagate,
        influence_scale=1.0
    )
    
    # Load influence-based models immediately after unlearning
    influence_models = {}
    for round_result in influence_results.get("unlearned_rounds", []):
        if round_result["status"] == "success":
            round_num = round_result["round"]
            # Load directly using the unlearner's method
            influence_params = influence_unlearner._load_aggregated_model(round_num)
            if influence_params:
                influence_models[round_num] = influence_params
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle dataset name (strip _CLASS_VERTICAL suffix if present)
    base_dataset = dataset_name.split("_")[0] if "_" in dataset_name else dataset_name
    
    if base_dataset == "CUSTOM":
        if num_classes is None or num_channels is None or img_size is None:
            print("  Warning: Skipping evaluation - missing dataset parameters for CUSTOM dataset")
            evaluate = False
            net = None
            testloader = None
        else:
            net = create_model(model_name, base_dataset,
                              num_classes=num_classes, num_channels=num_channels, img_size=img_size).to(device)
            _, testloader = load_data(
                base_dataset,
                num_clients=1,
                batch_size=32,
                dataset_path=dataset_path,
                img_size=img_size,
                num_channels=num_channels
            )
    else:
        net = create_model(model_name, base_dataset).to(device)
        _, testloader = load_data(base_dataset, num_clients=1, batch_size=32)
    
    comparison_data = []
    
    for round_num in affected_rounds:
        print(f"\nRound {round_num:04d}:")
        print("-" * 60)
        
        round_data = {"round": round_num}
        
        # Original model
        if round_num in original_models and evaluate and net is not None and testloader is not None:
            set_parameters(net, original_models[round_num])
            orig_loss, orig_acc = test(net, testloader, device)
            round_data["original"] = {"loss": orig_loss, "accuracy": orig_acc}
            print(f"  Original:        Loss={orig_loss:.4f}, Accuracy={orig_acc:.4f}")
        
        # Gradient-based
        if round_num in gradient_models and evaluate and net is not None and testloader is not None:
            set_parameters(net, gradient_models[round_num])
            grad_loss, grad_acc = test(net, testloader, device)
            round_data["gradient"] = {"loss": grad_loss, "accuracy": grad_acc}
            print(f"  Gradient-based:  Loss={grad_loss:.4f}, Accuracy={grad_acc:.4f}")
            if round_num in original_models and "original" in round_data:
                loss_diff = grad_loss - round_data["original"]["loss"]
                acc_diff = grad_acc - round_data["original"]["accuracy"]
                print(f"    Δ from original: Loss={loss_diff:+.4f}, Accuracy={acc_diff:+.4f}")
        
        # Influence-based
        if round_num in influence_models and evaluate and net is not None and testloader is not None:
            set_parameters(net, influence_models[round_num])
            inf_loss, inf_acc = test(net, testloader, device)
            round_data["influence"] = {"loss": inf_loss, "accuracy": inf_acc}
            print(f"  Influence-based: Loss={inf_loss:.4f}, Accuracy={inf_acc:.4f}")
            if round_num in original_models and "original" in round_data:
                loss_diff = inf_loss - round_data["original"]["loss"]
                acc_diff = inf_acc - round_data["original"]["accuracy"]
                print(f"    Δ from original: Loss={loss_diff:+.4f}, Accuracy={acc_diff:+.4f}")
        
        # Compare gradient vs influence
        if round_num in gradient_models and round_num in influence_models:
            # Calculate parameter differences
            param_diff_norms = []
            for i in range(len(gradient_models[round_num])):
                diff = np.array(gradient_models[round_num][i]) - np.array(influence_models[round_num][i])
                param_diff_norms.append(np.linalg.norm(diff))
            
            avg_param_diff = np.mean(param_diff_norms)
            max_param_diff = np.max(param_diff_norms)
            
            print(f"  Parameter difference (Gradient vs Influence):")
            print(f"    Average norm: {avg_param_diff:.6f}")
            print(f"    Max norm: {max_param_diff:.6f}")
            
            if evaluate and "gradient" in round_data and "influence" in round_data:
                loss_diff = round_data["gradient"]["loss"] - round_data["influence"]["loss"]
                acc_diff = round_data["gradient"]["accuracy"] - round_data["influence"]["accuracy"]
                print(f"  Performance difference:")
                print(f"    Loss difference: {loss_diff:+.4f}")
                print(f"    Accuracy difference: {acc_diff:+.4f}")
        
        comparison_data.append(round_data)
    
    print(f"\n{'='*60}\n")
    
    return {
        "client_id": client_id,
        "affected_rounds": affected_rounds,
        "comparison_data": comparison_data,
        "gradient_results": gradient_results,
        "influence_results": influence_results
    }


def main():
    parser = argparse.ArgumentParser(description="Manage client withdrawals and recalculate models")
    parser.add_argument(
        "command",
        choices=["withdraw", "restore", "history", "list", "recalculate", "unlearn", "compare"],
        help="Command to execute"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="Dataset name (MNIST, CIFAR10, FashionMNIST, CUSTOM, CUSTOM_CLASS_VERTICAL, etc.)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to custom dataset (required for CUSTOM datasets)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (required for CUSTOM datasets)",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Number of channels (required for CUSTOM datasets)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Image size (required for CUSTOM datasets)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simplenet",
        help="Model architecture (simplenet, resnet18)",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=None,
        help="Client ID (for withdraw, restore, or history)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Round number (for history or recalculate)",
    )
    parser.add_argument(
        "--from-round",
        type=int,
        default=None,
        help="Starting round for recalculate",
    )
    parser.add_argument(
        "--to-round",
        type=int,
        default=None,
        help="Ending round for recalculate",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Reason for withdrawal",
    )
    parser.add_argument(
        "--no-propagate",
        action="store_true",
        help="Don't propagate unlearning to subsequent rounds (for unlearn command)",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Don't evaluate unlearned models (for unlearn command)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="gradient",
        choices=["gradient", "influence"],
        help="Unlearning algorithm to use (for unlearn command)",
    )
    parser.add_argument(
        "--damping-factor",
        type=float,
        default=0.01,
        help="Damping factor for influence-based algorithm (default 0.01)",
    )
    parser.add_argument(
        "--influence-scale",
        type=float,
        default=1.0,
        help="Influence scale factor for influence-based algorithm (default 1.0)",
    )
    
    args = parser.parse_args()
    
    # Initialize database
    db = ContributionDB(dataset_name=args.dataset)
    
    if args.command == "withdraw":
        if args.client_id is None:
            print("Error: --client-id is required for withdraw command")
            return
        withdraw_client(db, args.client_id, args.reason)
    
    elif args.command == "restore":
        if args.client_id is None:
            print("Error: --client-id is required for restore command")
            return
        restore_client(db, args.client_id)
    
    elif args.command == "history":
        show_history(db, client_id=args.client_id, round_num=args.round)
    
    elif args.command == "list":
        list_withdrawals(db)
    
    elif args.command == "recalculate":
        recalculate_models(
            db, 
            args.dataset, 
            from_round=args.from_round, 
            to_round=args.to_round,
            model_name=args.model,
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            img_size=args.img_size,
            dataset_path=args.dataset_path
        )
    
    elif args.command == "unlearn":
        if args.client_id is None:
            print("Error: --client-id is required for unlearn command")
            return
        unlearn_client(
            db, 
            args.dataset, 
            args.client_id,
            algorithm=args.algorithm,
            propagate=not args.no_propagate,
            evaluate=not args.no_evaluate,
            model_name=args.model,
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            img_size=args.img_size,
            dataset_path=args.dataset_path,
            damping_factor=args.damping_factor,
            influence_scale=args.influence_scale
        )
    
    elif args.command == "compare":
        if args.client_id is None:
            print("Error: --client-id is required for compare command")
            return
        compare_unlearning_algorithms(
            db,
            args.dataset,
            args.client_id,
            propagate=not args.no_propagate,
            evaluate=not args.no_evaluate,
            model_name=args.model,
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            img_size=args.img_size,
            dataset_path=args.dataset_path
        )


if __name__ == "__main__":
    main()

