"""Script to view and analyze contributions in the database.
VISUALIZE IN TIMES NEW ROMAN

"""
import json
import argparse
from pathlib import Path
from contributions_db import ContributionDB


def main():
    parser = argparse.ArgumentParser(description="View contributions database")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "CIFAR10", "FashionMNIST"],
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Filter by specific round number",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=None,
        help="Filter by specific client ID",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show overall statistics",
    )
    
    args = parser.parse_args()
    
    # Initialize database
    db = ContributionDB(dataset_name=args.dataset)
    
    if args.stats:
        # Show statistics
        stats = db.get_statistics()
        print("\n" + "="*60)
        print(f"CONTRIBUTION STATISTICS - {args.dataset}")
        print("="*60)
        print(f"Total Contributions: {stats['total_contributions']}")
        print(f"Total Rounds: {stats['total_rounds']}")
        print(f"Unique Clients: {len(stats['clients_participated'])}")
        print(f"Clients: {sorted(stats['clients_participated'])}")
        print("\nPer Round Breakdown:")
        for round_num in sorted(stats['rounds'].keys()):
            round_info = stats['rounds'][round_num]
            print(f"  Round {round_num:04d}: {round_info['contributions']} contributions from {len(round_info['clients'])} clients")
            print(f"    Clients: {sorted(round_info['clients'])}")
        print("="*60 + "\n")
    else:
        # List contributions
        contributions = db.list_contributions(round_num=args.round)
        
        if args.client_id is not None:
            contributions = [c for c in contributions if c['client_id'] == args.client_id]
        
        if not contributions:
            print(f"No contributions found for dataset: {args.dataset}")
            if args.round is not None:
                print(f"Round: {args.round}")
            if args.client_id is not None:
                print(f"Client ID: {args.client_id}")
            return
        
        print("\n" + "="*60)
        print(f"CONTRIBUTIONS - {args.dataset}")
        if args.round is not None:
            print(f"Round: {args.round}")
        if args.client_id is not None:
            print(f"Client ID: {args.client_id}")
        print("="*60)
        
        for contrib in contributions:
            print(f"\nRound {contrib['round']:04d}, Client {contrib['client_id']:04d}")
            print(f"  Timestamp: {contrib['timestamp']}")
            print(f"  Training Samples: {contrib['num_samples']}")
            print(f"  Parameters: {contrib['parameters_count']} layers")
            if contrib.get('metrics'):
                print(f"  Metrics:")
                for key, value in contrib['metrics'].items():
                    print(f"    {key}: {value}")
            print(f"  Parameters File: {contrib['parameters_file']}")
        
        print("\n" + "="*60)
        print(f"Total: {len(contributions)} contributions")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()

