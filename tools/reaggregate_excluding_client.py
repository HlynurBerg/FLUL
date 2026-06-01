"""
Re-aggregate contributions excluding specific clients.

This script loads existing contributions and recalculates aggregated models
as if the excluded clients never participated. This allows comparison with
unlearned models to verify unlearning correctness.
"""

import argparse
from pathlib import Path
import torch
from contributions_db import ContributionDB
from model import create_model, set_parameters
from utils import load_data, test


def main():
    parser = argparse.ArgumentParser(
        description="Re-aggregate contributions excluding specific clients"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Original dataset name (e.g., CUSTOM_CLASS_VERTICAL)",
    )
    parser.add_argument(
        "--exclude-clients",
        type=str,
        required=True,
        help="Comma-separated list of client IDs to exclude (e.g., '3' or '2,3')",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Suffix for output database (e.g., 'RETRAINED_012'). Default: auto-generated from excluded clients",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simplenet",
        help="Model architecture (for evaluation)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (required for CUSTOM dataset)",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Number of channels (required for CUSTOM dataset)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Image size (required for CUSTOM dataset)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to custom dataset root (for evaluation)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate recalculated models and save metrics",
    )
    args = parser.parse_args()
    
    # Parse excluded clients
    try:
        excluded_client_ids = {int(cid.strip()) for cid in args.exclude_clients.split(",")}
        excluded_str = "_".join(sorted(str(cid) for cid in excluded_client_ids))
    except ValueError as e:
        print(f"ERROR: Invalid --exclude-clients format: {e}")
        return
    
    # Generate output suffix if not provided
    if args.output_suffix is None:
        args.output_suffix = f"EXCLUDED_{excluded_str}"
    
    print("=" * 60)
    print("Re-aggregating Contributions (Excluding Clients)")
    print("=" * 60)
    print(f"Original dataset: {args.dataset}")
    print(f"Excluding clients: {sorted(excluded_client_ids)}")
    print(f"Output database: {args.dataset}_{args.output_suffix}")
    print("=" * 60)
    
    # Load original contribution database
    original_db = ContributionDB(dataset_name=args.dataset)
    
    if not original_db.contributions_dir.exists():
        print(f"ERROR: Original contributions not found at {original_db.contributions_dir}")
        return
    
    # Get all rounds - try statistics first, then scan directory
    stats = original_db.get_statistics()
    all_rounds_from_stats = sorted(stats.get("rounds", {}).keys())
    
    # Also scan directory to make sure we get all rounds (in case stats are incomplete)
    all_rounds_from_dir = []
    for d in original_db.contributions_dir.iterdir():
        if d.is_dir() and d.name.startswith("round_"):
            try:
                rnd = int(d.name.split("_")[1])
                all_rounds_from_dir.append(rnd)
            except (ValueError, IndexError):
                continue
    all_rounds_from_dir = sorted(set(all_rounds_from_dir))
    
    # Use the union of both (directory scan is more reliable)
    all_rounds = sorted(set(all_rounds_from_stats + all_rounds_from_dir))
    
    if not all_rounds:
        print("ERROR: No rounds found in original contributions")
        print(f"  Checked directory: {original_db.contributions_dir}")
        print(f"  Statistics had: {len(all_rounds_from_stats)} rounds")
        print(f"  Directory scan found: {len(all_rounds_from_dir)} rounds")
        return
    
    print(f"Found {len(all_rounds)} rounds: {min(all_rounds)} to {max(all_rounds)}")
    if len(all_rounds_from_stats) != len(all_rounds):
        print(f"  (Statistics had {len(all_rounds_from_stats)} rounds, directory scan found {len(all_rounds_from_dir)})")
    
    # Create output database
    output_db_name = f"{args.dataset}_{args.output_suffix}"
    output_db = ContributionDB(dataset_name=output_db_name)
    print(f"Output database: {output_db.contributions_dir}")
    
    # Copy contributions from non-excluded clients
    print("\nCopying contributions from non-excluded clients...")
    all_contributions = original_db.list_contributions(exclude_withdrawn=False)
    
    copied_count = 0
    for contrib in all_contributions:
        client_id = contrib["client_id"]
        if client_id in excluded_client_ids:
            continue
        
        round_num = contrib["round"]
        
        # Load parameters from original database
        round_dir = original_db.contributions_dir / f"round_{round_num:04d}"
        client_dir = round_dir / f"client_{client_id:04d}"
        
        if not client_dir.exists():
            continue
        
        # Find the parameters file
        params_file_name = contrib.get("parameters_file", "")
        if not params_file_name:
            # Try to find any params file
            params_files = list(client_dir.glob("*_params.pkl"))
            if not params_files:
                continue
            params_file = params_files[0]
            params_file_name = params_file.name
        else:
            params_file = client_dir / params_file_name
            if not params_file.exists():
                # Try to find any params file
                params_files = list(client_dir.glob("*_params.pkl"))
                if not params_files:
                    continue
                params_file = params_files[0]
                params_file_name = params_file.name
        
        # Load parameters
        import pickle
        with open(params_file, 'rb') as f:
            parameters = pickle.load(f)
        
        # Save to output database
        output_db.save_contribution(
            round_num=round_num,
            client_id=client_id,
            parameters=parameters,
            num_samples=contrib["num_samples"],
            metrics=contrib.get("metrics", {})
        )
        copied_count += 1
    
    print(f"Copied {copied_count} contributions from non-excluded clients")
    
    # Recalculate aggregated models for each round
    print("\nRecalculating aggregated models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare for evaluation if requested
    test_loader = None
    if args.evaluate:
        # Normalize dataset name for model creation
        base_dataset = args.dataset.split("_")[0] if "_" in args.dataset else args.dataset
        if "CUSTOM" in base_dataset.upper() or args.dataset.endswith("_CLASS_VERTICAL"):
            base_dataset = "CUSTOM"
        
        # Create model for evaluation
        if base_dataset == "CIFAR10":
            net = create_model(args.model, base_dataset, num_classes=10, num_channels=3, img_size=32).to(device)
        elif base_dataset in ("MNIST", "FashionMNIST"):
            net = create_model(args.model, base_dataset, num_classes=10, num_channels=1, img_size=28).to(device)
        elif base_dataset == "CUSTOM":
            if args.num_classes is None or args.num_channels is None or args.img_size is None:
                print("WARNING: Cannot evaluate without num_classes, num_channels, and img_size")
                args.evaluate = False
            else:
                net = create_model(
                    args.model, base_dataset,
                    num_classes=args.num_classes,
                    num_channels=args.num_channels,
                    img_size=args.img_size
                ).to(device)
                # Load test data
                _, test_loader = load_data(
                    base_dataset,
                    num_clients=1,
                    batch_size=128,
                    partition_type="horizontal",
                    dataset_path=args.dataset_path,
                    img_size=args.img_size,
                    num_channels=args.num_channels
                )
        else:
            print(f"WARNING: Cannot evaluate for dataset {args.dataset}")
            args.evaluate = False
    
    recalculated_count = 0
    failed_rounds = []
    for round_num in all_rounds:
        print(f"  Round {round_num}...", end=" ", flush=True)
        
        # Check if we have contributions for this round in the output DB
        round_contribs = output_db.list_contributions(round_num=round_num, exclude_withdrawn=False)
        if not round_contribs:
            print(f"SKIPPED (no contributions copied for this round)")
            failed_rounds.append(round_num)
            continue
        
        # Recalculate aggregated model using only non-excluded clients
        # (We already excluded them by not copying their contributions)
        recalculated_params = output_db.recalculate_aggregated_model(
            round_num=round_num,
            exclude_withdrawn=False  # No need to exclude - they're not in the database
        )
        
        if recalculated_params is None:
            print("FAILED (recalculation failed)")
            failed_rounds.append(round_num)
            continue
        
        # Evaluate if requested
        metrics = {}
        if args.evaluate and test_loader is not None:
            set_parameters(net, recalculated_params)
            loss, accuracy = test(net, test_loader, device)
            metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy)
            }
        
        # Save recalculated aggregated model
        output_db.save_aggregated_model(
            round_num=round_num,
            parameters=recalculated_params,
            metrics=metrics,
            suffix=None  # No suffix - these are the "normal" aggregated models
        )
        
        recalculated_count += 1
        if args.evaluate:
            print(f"✓ (acc: {metrics.get('accuracy', 0):.4f})")
        else:
            print("✓")
    
    print(f"\nRecalculated {recalculated_count} aggregated models")
    if failed_rounds:
        print(f"  WARNING: {len(failed_rounds)} rounds failed: {failed_rounds[:10]}{'...' if len(failed_rounds) > 10 else ''}")
    print(f"\nOutput saved to: {output_db.contributions_dir}")
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    
    if args.evaluate:
        print(f"\nTo visualize:")
        print(f"  python visualize_metrics.py --dataset {output_db_name} --model {args.model} \\")
        if args.num_classes:
            print(f"    --num-classes {args.num_classes} --num-channels {args.num_channels} --img-size {args.img_size} \\")
        if args.dataset_path:
            print(f"    --dataset-path {args.dataset_path} \\")
        print(f"    --model-variant retrained --per-class")


if __name__ == "__main__":
    main()

