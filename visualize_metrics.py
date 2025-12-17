"""Visualize per-client (feature-slice) metrics over rounds. 

Reads from the contributions database structure created by server-side
ContributionDB and generates line plots for accuracy/loss per client
and the aggregated model.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _load_aggregated_round_metrics(contrib_root: Path, model_variant: str = "original") -> Dict[int, Dict[str, float]]:
    """Return {round_num: metrics_dict} for aggregated models.
    
    Args:
        contrib_root: Root directory of contributions
        model_variant: Which model variant to load:
            - "original": Regular aggregated models (default)
            - "unlearned_gradient": Gradient-based unlearned models
            - "unlearned_influence": Influence-based unlearned models
            - "retrained": Re-aggregated models (excluding some clients)
    """
    round_to_metrics: Dict[int, Dict[str, float]] = {}
    rounds_found = []
    
    for d in contrib_root.iterdir():
        if not d.is_dir() or not d.name.startswith("round_"):
            continue
        try:
            rnd = int(d.name.split("_")[1])
            rounds_found.append(rnd)
        except Exception:
            continue
        
        # Filter metadata files based on model variant
        if model_variant == "original":
            # Original aggregated models (no unlearned suffix)
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned" not in p.name]
        elif model_variant == "unlearned_gradient":
            # Gradient-based unlearned models
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned_gradient" in p.name]
        elif model_variant == "unlearned_influence":
            # Influence-based unlearned models
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned_influence" in p.name]
        elif model_variant == "retrained":
            # Retrained/re-aggregated models (no unlearned suffix, but from different database)
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned" not in p.name]
        else:
            metas = []
        
        if not metas:
            if model_variant == "retrained":
                print(f"  WARNING: Round {rnd}: No aggregated metadata files found")
            continue
        
        latest = max(metas, key=lambda p: p.stat().st_mtime)
        with open(latest, "r") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        round_to_metrics[rnd] = metrics
        
        # Debug output for retrained models
        if model_variant == "retrained" and not metrics:
            print(f"  WARNING: Round {rnd}: Metadata file found but metrics are empty")
            print(f"    File: {latest.name}")
            print(f"    Metadata keys: {list(data.keys())}")
    
    if model_variant == "retrained":
        print(f"  Found {len(rounds_found)} rounds total, {len(round_to_metrics)} with metrics")
        if rounds_found:
            print(f"  Rounds: {sorted(rounds_found)}")
    
    return round_to_metrics


def _load_client_contributions(contrib_root: Path) -> List[Dict]:
    """Flatten all client contribution metadata across rounds."""
    contributions: List[Dict] = []
    for d in contrib_root.iterdir():
        if not d.is_dir() or not d.name.startswith("round_"):
            continue
        for client_dir in d.iterdir():
            if client_dir.is_dir() and client_dir.name.startswith("client_"):
                for meta in client_dir.glob("*_metadata.json"):
                    with open(meta, "r") as f:
                        contributions.append(json.load(f))
    # Sort by round then timestamp
    contributions.sort(key=lambda x: (x.get("round", 0), x.get("timestamp", "")))
    return contributions


def _collect_series(contributions: List[Dict], metric_key: str) -> Dict[int, List[Tuple[int, float]]]:
    """Return {client_id: [(round, value), ...]} for the given metric key."""
    series: Dict[int, List[Tuple[int, float]]] = {}
    for c in contributions:
        client_id = int(c.get("client_id", -1))
        rnd = int(c.get("round", -1))
        metrics = c.get("metrics", {})
        if metric_key in metrics:
            try:
                val = float(metrics[metric_key])
            except Exception:
                continue
            series.setdefault(client_id, []).append((rnd, val))
    # Sort each client's series by round
    for cid in series:
        series[cid].sort(key=lambda x: x[0])
    return series


def _evaluate_aggregated_models(
    contrib_root: Path,
    dataset: str,
    model_name: str,
    num_classes: Optional[int],
    num_channels: Optional[int],
    img_size: Optional[int],
    dataset_path: Optional[str],
    model_variant: str,
    metric_key: str
) -> Dict[int, float]:
    """Evaluate aggregated models and return metrics.
    
    Returns {round_num: metric_value} for the specified metric.
    """
    import torch
    from model import create_model, set_parameters
    from utils import load_data, test
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize dataset name
    base_dataset = dataset.split("_")[0] if "_" in dataset else dataset
    if "CUSTOM" in base_dataset.upper() or dataset.endswith("_CLASS_VERTICAL"):
        base_dataset = "CUSTOM"
    
    # Create model
    if base_dataset == "CIFAR10":
        net = create_model(model_name, base_dataset, num_classes=10, num_channels=3, img_size=32).to(device)
    elif base_dataset in ("MNIST", "FashionMNIST"):
        net = create_model(model_name, base_dataset, num_classes=10, num_channels=1, img_size=28).to(device)
    elif base_dataset == "CUSTOM":
        if num_classes is None or num_channels is None or img_size is None:
            raise ValueError("For CUSTOM dataset, num_classes, num_channels, and img_size must be provided")
        net = create_model(model_name, base_dataset, num_classes=num_classes, num_channels=num_channels, img_size=img_size).to(device)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Load test data
    _, test_loader = load_data(
        base_dataset,
        num_clients=1,
        batch_size=128,
        partition_type="horizontal",
        dataset_path=dataset_path,
        img_size=img_size,
        num_channels=num_channels
    )
    
    results = {}
    
    for d in sorted(contrib_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or not d.name.startswith("round_"):
            continue
        try:
            rnd = int(d.name.split("_")[1])
        except Exception:
            continue
        
        # Filter metadata files based on model variant
        if model_variant == "original":
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned" not in p.name]
        elif model_variant == "unlearned_gradient":
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned_gradient" in p.name]
        elif model_variant == "unlearned_influence":
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned_influence" in p.name]
        elif model_variant == "retrained":
            # Retrained models (no unlearned suffix, but from different database)
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned" not in p.name]
        else:
            metas = []
        
        if not metas:
            continue
        
        latest = max(metas, key=lambda p: p.stat().st_mtime)
        with open(latest, "r") as f:
            data = json.load(f)
        params_file = d / data.get("parameters_file", "")
        if not params_file.exists():
            continue
        
        # Load parameters
        import pickle
        with open(params_file, "rb") as pf:
            params = pickle.load(pf)
        set_parameters(net, params)
        
        # Evaluate
        loss, accuracy = test(net, test_loader, device)
        
        if metric_key == "accuracy":
            results[rnd] = float(accuracy)
        elif metric_key == "loss":
            results[rnd] = float(loss)
    
    return results


def _evaluate_aggregated_per_class(
    dataset: str, 
    contrib_root: Path,
    num_classes: Optional[int] = None,
    num_channels: Optional[int] = None,
    img_size: Optional[int] = None,
    model_name: str = "simplenet",
    dataset_path: Optional[str] = None,
    model_variant: str = "original"
) -> Tuple[List[int], List[List[float]]]:
    """Compute per-class accuracies for each round's aggregated model.

    Returns (rounds, per_class_acc_list) where per_class_acc_list is a list of
    length num_rounds, each an array/list of length num_classes.
    """
    import torch
    from model import create_model, set_parameters
    from utils import load_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Handle dataset name - strip suffixes like _CLASS_VERTICAL for model creation
    base_dataset = dataset.split("_")[0] if "_" in dataset else dataset

    if base_dataset == "CIFAR10":
        net = create_model(model_name, base_dataset, num_classes=10, num_channels=3, img_size=32).to(device)
        num_classes = 10
    elif base_dataset in ("MNIST", "FashionMNIST"):
        net = create_model(model_name, base_dataset, num_classes=10, num_channels=1, img_size=28).to(device)
        num_classes = 10
    elif base_dataset == "CUSTOM":
        if num_classes is None or num_channels is None or img_size is None:
            raise ValueError("For CUSTOM dataset, num_classes, num_channels, and img_size must be provided")
        net = create_model(model_name, base_dataset, num_classes=num_classes, num_channels=num_channels, img_size=img_size).to(device)
    else:
        raise ValueError(f"Unsupported dataset for per-class eval: {dataset}")

    # Get an unmasked test loader - use base_dataset for load_data
    _, test_loader = load_data(
        base_dataset, 
        num_clients=1, 
        batch_size=128, 
        partition_type="horizontal",
        dataset_path=dataset_path,
        img_size=img_size,
        num_channels=num_channels
    )

    rounds = []
    per_class_acc_over_time: List[List[float]] = []

    for d in sorted(contrib_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or not d.name.startswith("round_"):
            continue
        try:
            rnd = int(d.name.split("_")[1])
        except Exception:
            continue
        # Filter metadata files based on model variant
        if model_variant == "original":
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned" not in p.name]
        elif model_variant == "unlearned_gradient":
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned_gradient" in p.name]
        elif model_variant == "unlearned_influence":
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned_influence" in p.name]
        elif model_variant == "retrained":
            # Retrained models (no unlearned suffix, but from different database)
            metas = [p for p in d.glob("*_aggregated_*_metadata.json") 
                    if "unlearned" not in p.name]
        else:
            metas = []
        
        if not metas:
            continue
        latest = max(metas, key=lambda p: p.stat().st_mtime)
        with open(latest, "r") as f:
            data = json.load(f)
        params_file = d / data.get("parameters_file", "")
        if not params_file.exists():
            continue
        # Load parameters
        import pickle
        with open(params_file, "rb") as pf:
            params = pickle.load(pf)
        set_parameters(net, params)

        # Evaluate per-class
        net.eval()
        correct = [0 for _ in range(num_classes)]
        total = [0 for _ in range(num_classes)]
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                for cls in range(num_classes):
                    mask = labels == cls
                    total[cls] += mask.sum().item()
                    correct[cls] += (predicted[mask] == labels[mask]).sum().item()
        acc = [ (correct[c] / total[c]) if total[c] > 0 else 0.0 for c in range(num_classes) ]
        rounds.append(rnd)
        per_class_acc_over_time.append(acc)
        
        # Debug output for first round
        if len(rounds) == 1 and model_variant == "unlearned_influence":
            print(f"  Round {rnd}: Per-class accuracies: {[f'{a:.4f}' for a in acc]}")

    return rounds, per_class_acc_over_time


def main():
    parser = argparse.ArgumentParser(description="Visualize federated metrics per client")
    parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset name (e.g., MNIST, CIFAR10, CUSTOM, CUSTOM_CLASS_VERTICAL)")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Directory to save plots")
    parser.add_argument("--metrics", type=str, default="local_accuracy,local_loss,accuracy,loss", help="Comma-separated metric keys to plot")
    parser.add_argument("--per-class", action="store_true", help="Additionally compute and plot per-class accuracy for aggregated models")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (required for CUSTOM dataset)")
    parser.add_argument("--num-channels", type=int, default=None, help="Number of channels (required for CUSTOM dataset)")
    parser.add_argument("--img-size", type=int, default=None, help="Image size (required for CUSTOM dataset)")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to custom dataset root (required for CUSTOM dataset with --per-class)")
    parser.add_argument("--model", type=str, default="simplenet", help="Model architecture used (for per-class evaluation)")
    parser.add_argument("--model-variant", type=str, default="original", 
                       choices=["original", "unlearned_gradient", "unlearned_influence", "retrained"],
                       help="Which model variant to visualize: original, unlearned_gradient, unlearned_influence, or retrained")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        # Set Times New Roman as the default font family
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.serif'] = ['Times New Roman']
        # Also set font for all text elements
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.serif'] = ['Times New Roman']
    except Exception:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    # Handle retrained models - they're in a different database
    if args.model_variant == "retrained":
        # Retrained/re-aggregated models are in a database with RETRAINED or EXCLUDED suffix
        # Try common suffixes
        retrain_suffixes = ["RETRAINED_012", "RETRAINED", "EXCLUDED_3", "EXCLUDED"]
        contrib_root = None
        for suffix in retrain_suffixes:
            test_path = Path("contributions") / f"{args.dataset}_{suffix}"
            if test_path.exists():
                contrib_root = test_path
                print(f"Found retrained contributions at: {contrib_root}")
                break
        
        if contrib_root is None:
            # Try to find any contribution directory with RETRAINED or EXCLUDED in the name
            base_contrib = Path("contributions")
            if base_contrib.exists():
                dataset_base = args.dataset.split("_")[0]
                for subdir in base_contrib.iterdir():
                    if subdir.is_dir():
                        name = subdir.name
                        # Check if it contains RETRAINED or EXCLUDED and matches dataset
                        if ("RETRAINED" in name or "EXCLUDED" in name) and dataset_base in name:
                            contrib_root = subdir
                            print(f"Found retrained contributions at: {contrib_root}")
                            break
        
        if contrib_root is None:
            print(f"No retrained contributions found. Expected: contributions/{args.dataset}_RETRAINED_* or {args.dataset}_EXCLUDED_*")
            print("Available directories:")
            if Path("contributions").exists():
                for subdir in Path("contributions").iterdir():
                    if subdir.is_dir():
                        print(f"  - {subdir.name}")
            return
    else:
        contrib_root = Path("contributions") / args.dataset
    
    if not contrib_root.exists():
        print(f"No contributions found at {contrib_root}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    aggregated_by_round = _load_aggregated_round_metrics(contrib_root, model_variant=args.model_variant)
    contributions = _load_client_contributions(contrib_root)
    
    # Debug: Print what we loaded
    if args.model_variant != "original":
        print(f"\nLoaded {len(aggregated_by_round)} rounds for {args.model_variant} model")
        if aggregated_by_round:
            sample_round = min(aggregated_by_round.keys())
            sample_metrics = aggregated_by_round[sample_round]
            print(f"  Sample round {sample_round} metrics: {list(sample_metrics.keys())}")
            if "accuracy" in sample_metrics:
                print(f"  Accuracy value: {sample_metrics['accuracy']}")
            if "loss" in sample_metrics:
                print(f"  Loss value: {sample_metrics['loss']}")

    metric_keys = [m.strip() for m in args.metrics.split(",") if m.strip()]

    for metric_key in metric_keys:
        series_by_client = _collect_series(contributions, metric_key)
        if not series_by_client and metric_key not in {"accuracy", "loss"}:
            # Skip if nothing to plot
            continue

        plt.figure(figsize=(8, 5))
        # Plot per-client lines
        for client_id, pts in sorted(series_by_client.items()):
            if not pts:
                continue
            xs = [r for r, _ in pts]
            ys = [v for _, v in pts]
            plt.plot(xs, ys, marker='o', label=f"Client {client_id}")

        # If metric is server-level (accuracy/loss) and aggregated available, add it
        if metric_key in {"accuracy", "loss"} and aggregated_by_round:
            xs_agg = sorted(aggregated_by_round.keys())
            ys_agg = []
            
            # Check if we need to evaluate models (metrics missing or invalid)
            need_evaluation = False
            stored_metrics_count = 0
            for r in xs_agg:
                metric_value = aggregated_by_round[r].get(metric_key)
                if metric_value is None or metric_value == {}:
                    # Empty dict or None
                    need_evaluation = True
                elif isinstance(metric_value, (int, float)):
                    if metric_value != metric_value:  # NaN check
                        need_evaluation = True
                    elif metric_key == "accuracy" and metric_value == 0.0 and len(xs_agg) > 1:
                        # Suspicious zero accuracy (but allow if it's the only round)
                        need_evaluation = True
                    else:
                        stored_metrics_count += 1
                else:
                    stored_metrics_count += 1
            
            # For retrained models, always evaluate if we have missing metrics
            # For other variants, evaluate if more than 50% are missing
            should_evaluate = need_evaluation or (
                args.model_variant != "original" and 
                (stored_metrics_count < len(xs_agg) * 0.5 or args.model_variant == "retrained")
            )
            
            if should_evaluate:
                # Evaluate models on-the-fly
                print(f"\nEvaluating {args.model_variant} models for {metric_key} ({len(xs_agg)} rounds)...")
                print(f"  Stored metrics: {stored_metrics_count}/{len(xs_agg)}")
                try:
                    evaluated_metrics = _evaluate_aggregated_models(
                        contrib_root,
                        args.dataset,
                        args.model,
                        args.num_classes,
                        args.num_channels,
                        args.img_size,
                        args.dataset_path,
                        args.model_variant,
                        metric_key
                    )
                    print(f"  Evaluated {len(evaluated_metrics)} rounds")
                    # Use evaluated metrics, fall back to stored if available
                    for r in xs_agg:
                        if r in evaluated_metrics:
                            ys_agg.append(float(evaluated_metrics[r]))
                        else:
                            # Fall back to stored metric if available
                            stored_val = aggregated_by_round[r].get(metric_key)
                            if stored_val is not None and isinstance(stored_val, (int, float)):
                                try:
                                    val = float(stored_val)
                                    if val == val:  # Not NaN
                                        ys_agg.append(val)
                                    else:
                                        ys_agg.append(float('nan'))
                                except (ValueError, TypeError):
                                    ys_agg.append(float('nan'))
                            else:
                                ys_agg.append(float('nan'))
                except Exception as e:
                    print(f"  ERROR: Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to stored metrics
                    for r in xs_agg:
                        stored_val = aggregated_by_round[r].get(metric_key)
                        if stored_val is not None and isinstance(stored_val, (int, float)):
                            try:
                                val = float(stored_val)
                                if val == val:  # Not NaN
                                    ys_agg.append(val)
                                else:
                                    ys_agg.append(float('nan'))
                            except (ValueError, TypeError):
                                ys_agg.append(float('nan'))
                        else:
                            ys_agg.append(float('nan'))
            else:
                # Use stored metrics
                ys_agg = []
                for r in xs_agg:
                    stored_val = aggregated_by_round[r].get(metric_key)
                    if stored_val is not None and isinstance(stored_val, (int, float)):
                        try:
                            val = float(stored_val)
                            if val == val:  # Not NaN
                                ys_agg.append(val)
                            else:
                                ys_agg.append(float('nan'))
                        except (ValueError, TypeError):
                            ys_agg.append(float('nan'))
                    else:
                        ys_agg.append(float('nan'))
            
            variant_label = args.model_variant.replace("_", " ").title()
            plt.plot(xs_agg, ys_agg, color='black', linewidth=2, label=f"Aggregated ({variant_label})")

        # Set font for all text elements
        plt.title(f"{args.dataset} - {metric_key} over rounds", fontfamily='Times New Roman', fontsize=14)
        plt.xlabel("Round", fontfamily='Times New Roman', fontsize=12)
        plt.ylabel(metric_key, fontfamily='Times New Roman', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        legend = plt.legend(fontsize=10)
        for text in legend.get_texts():
            text.set_fontfamily('Times New Roman')
        # Set font for tick labels
        for label in plt.gca().get_xticklabels():
            label.set_fontfamily('Times New Roman')
        for label in plt.gca().get_yticklabels():
            label.set_fontfamily('Times New Roman')
        outfile = output_dir / f"{args.dataset}_{metric_key}.png"
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"Saved {outfile}")

    # Optional: per-class accuracy curves for aggregated model
    if args.per_class:
        print(f"\nGenerating per-class accuracy visualization for {args.model_variant} model...")
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            print("ERROR: matplotlib not available for per-class visualization")
        else:
            try:
                print(f"  Evaluating per-class accuracy for {args.model_variant} models...")
                rounds, per_class = _evaluate_aggregated_per_class(
                    args.dataset, 
                    contrib_root,
                    num_classes=args.num_classes,
                    num_channels=args.num_channels,
                    img_size=args.img_size,
                    model_name=args.model,
                    dataset_path=args.dataset_path,
                    model_variant=args.model_variant
                )
                print(f"  Evaluated {len(rounds)} rounds, {len(per_class[0]) if per_class else 0} classes")
                if rounds and per_class:
                    arr = np.array(per_class)  # shape: (T, C)
                    plt.figure(figsize=(10, 6))
                    for cls in range(arr.shape[1]):
                        plt.plot(rounds, arr[:, cls], marker='o', label=f"Class {cls}")
                    variant_label = args.model_variant.replace("_", " ").title()
                    plt.title(f"{args.dataset} - {variant_label} per-class accuracy over rounds", 
                             fontfamily='Times New Roman', fontsize=14)
                    plt.xlabel("Round", fontfamily='Times New Roman', fontsize=12)
                    plt.ylabel("Accuracy", fontfamily='Times New Roman', fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.3)
                    legend = plt.legend(ncol=2, fontsize=10)
                    for text in legend.get_texts():
                        text.set_fontfamily('Times New Roman')
                    # Set font for tick labels
                    for label in plt.gca().get_xticklabels():
                        label.set_fontfamily('Times New Roman')
                    for label in plt.gca().get_yticklabels():
                        label.set_fontfamily('Times New Roman')
                    variant_suffix = f"_{args.model_variant}" if args.model_variant != "original" else ""
                    outfile = output_dir / f"{args.dataset}_per_class_accuracy{variant_suffix}.png"
                    plt.tight_layout()
                    plt.savefig(outfile, dpi=300)
                    plt.close()
                    print(f"Saved {outfile}")
            except Exception as e:
                print(f"Error generating per-class plot: {e}")
                if "CUSTOM" in args.dataset.upper():
                    missing = []
                    if args.num_classes is None:
                        missing.append("--num-classes")
                    if args.num_channels is None:
                        missing.append("--num-channels")
                    if args.img_size is None:
                        missing.append("--img-size")
                    if args.dataset_path is None:
                        missing.append("--dataset-path")
                    if missing:
                        print(f"Note: For CUSTOM dataset, the following are required for per-class evaluation: {', '.join(missing)}")


if __name__ == "__main__":
    main()


