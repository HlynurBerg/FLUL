"""Flower server for federated learning."""
import argparse

import flwr as fl
import torch

from contributions_db import ContributionDB
from model import AVAILABLE_MODELS, create_model, get_parameters, set_parameters
from utils import load_data, test


def main():
    """Run Flower server."""
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "CIFAR10", "FashionMNIST", "CUSTOM"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simplenet",
        choices=list(AVAILABLE_MODELS),
        help="Model architecture to use",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to custom dataset root (expects train/ and val/ subfolders)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override input size (pixels). Required for custom datasets.",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Override number of channels. Required for custom datasets.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override classifier output classes. Required for custom datasets.",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds to run",
    )
    parser.add_argument(
        "--partition-type",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical", "class_vertical", "dirichlet"],
        help="Partition type (for contribution database naming)",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=None,
        help="Dirichlet concentration for partition_type=dirichlet (DB-naming only on the server; the per-client partitioning happens in client.py).",
    )
    parser.add_argument(
        "--primary-share",
        type=float,
        default=0.85,
        help="class_vertical primary-class share (DB-naming only on server; default 0.85).",
    )
    parser.add_argument(
        "--secondary-share",
        type=float,
        default=0.05,
        help="class_vertical secondary-class share (DB-naming only on server; default 0.05).",
    )
    parser.add_argument(
        "--exclude-clients",
        type=str,
        default=None,
        help="Comma-separated list of client IDs to exclude (e.g., '3' or '2,3'). Used for retraining with subset of clients.",
    )
    parser.add_argument(
        "--retrain-suffix",
        type=str,
        default=None,
        help="Suffix to add to contribution database name for retrained models (e.g., 'RETRAINED_012' to exclude client 3)",
    )
    parser.add_argument(
        "--init-weights-from-dataset",
        type=str,
        default=None,
        help="DB subdir under contributions/ to warm-start from (e.g., 'CIFAR10_UNLEARNED_class_pruning_p05_ft2'). Loads the newest aggregate at --init-weights-from-round and uses it as the initial global model.",
    )
    parser.add_argument(
        "--init-weights-from-round",
        type=int,
        default=None,
        help="Round number to load the warm-start aggregate from. Required when --init-weights-from-dataset is set.",
    )
    parser.add_argument(
        "--first-round-offset",
        type=int,
        default=0,
        help="Added to server_round when writing per-round contributions and aggregates. Used to preserve global round numbering when continuing FL training from a warm-started checkpoint (e.g., offset=K means server round 1 is saved as DB round K+1).",
    )
    args = parser.parse_args()
    
    # Load model and test data for server-side evaluation
    dataset_name = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log device information
    if torch.cuda.is_available():
        print(f"[Server] Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"[Server] CUDA device count: {torch.cuda.device_count()}")
        print(f"[Server] CUDA version: {torch.version.cuda}")
    else:
        print("[Server] CUDA not available, using CPU")
    print(f"[Server] Device: {device}")
    
    net = create_model(
        args.model,
        dataset_name,
        num_classes=args.num_classes,
        num_channels=args.num_channels,
        img_size=args.img_size,
    ).to(device)
    
    # Use smaller batch size for large images to avoid GPU memory issues
    batch_size = 8 if (args.img_size and args.img_size > 224) else 32
    print(f"[Server] Using batch size: {batch_size} for evaluation")
    
    _, testloader = load_data(
        dataset_name,
        num_clients=1,
        batch_size=batch_size,
        dataset_path=args.dataset_path,
        img_size=args.img_size,
        num_channels=args.num_channels,
    )
    
    # Initialize contribution database. Auto-suffix the DB name per partitioning
    # so different sweep configurations don't overwrite each other.
    if args.partition_type == "class_vertical":
        db_dataset_name = f"{dataset_name}_CLASS_VERTICAL"
        # Only append a share suffix when non-default values are used (preserves
        # backward compatibility with existing DBs at 0.85/0.05).
        if not (abs(args.primary_share - 0.85) < 1e-9 and abs(args.secondary_share - 0.05) < 1e-9):
            p_tag = f"{args.primary_share:.2f}".rstrip("0").rstrip(".")
            s_tag = f"{args.secondary_share:.2f}".rstrip("0").rstrip(".")
            db_dataset_name = f"{db_dataset_name}_p{p_tag}_s{s_tag}"
        print(f"[Server] Using class-based vertical partitioning - contributions saved separately as '{db_dataset_name}'")
    elif args.partition_type == "dirichlet":
        if args.dirichlet_alpha is None:
            raise ValueError("--dirichlet-alpha is required when --partition-type=dirichlet")
        a_tag = f"{args.dirichlet_alpha:g}".replace(".", "p")
        db_dataset_name = f"{dataset_name}_DIRICHLET_a{a_tag}"
        print(f"[Server] Using Dirichlet(α={args.dirichlet_alpha}) partitioning - contributions saved separately as '{db_dataset_name}'")
    else:
        db_dataset_name = dataset_name
    
    # Add retrain suffix if specified (for retraining with subset of clients)
    if args.retrain_suffix:
        db_dataset_name = f"{db_dataset_name}_{args.retrain_suffix}"
        print(f"[Server] Using retrain suffix - contributions saved as '{db_dataset_name}'")
    
    contribution_db = ContributionDB(dataset_name=db_dataset_name)
    print(f"[Server] Contribution database initialized: {contribution_db.contributions_dir}")

    # Warm-start: replace freshly-initialised net's params with the aggregate
    # from a prior run before saving initial_global_params. The MIA/replay layer
    # treats initial_global_params as the "start of round 1" model, so this also
    # serves as the source for client-side incoming weights.
    if args.init_weights_from_dataset is not None:
        if args.init_weights_from_round is None:
            raise ValueError("--init-weights-from-round is required when --init-weights-from-dataset is set")
        src_db = ContributionDB(dataset_name=args.init_weights_from_dataset)
        src_params = src_db.get_latest_aggregated_parameters(args.init_weights_from_round)
        if src_params is None:
            raise ValueError(
                f"No aggregate found at round {args.init_weights_from_round} in "
                f"'{args.init_weights_from_dataset}' (looked in {src_db.contributions_dir})"
            )
        set_parameters(net, src_params)
        print(
            f"[Server] Warm-started from {args.init_weights_from_dataset} round "
            f"{args.init_weights_from_round} (newest aggregate). "
            f"first_round_offset={args.first_round_offset}"
        )

    # Incoming weights for round 0 (replay / per-sample gradient recovery)
    contribution_db.save_initial_global_parameters(get_parameters(net))
    
    # Parse excluded clients
    excluded_client_ids = set()
    if args.exclude_clients:
        try:
            excluded_client_ids = {int(cid.strip()) for cid in args.exclude_clients.split(",")}
            print(f"[Server] Excluding clients: {sorted(excluded_client_ids)}")
        except ValueError as e:
            print(f"[Server] WARNING: Invalid --exclude-clients format: {e}. Ignoring.")
    
    def get_evaluate_fn():
        """Return function for server-side evaluation."""
        def evaluate(server_round, parameters, config):
            """Use the entire test set for evaluation."""
            set_parameters(net, parameters)
            
            # Debug: Comprehensive logging (first round and every 5 rounds)
            if server_round == 0 or server_round % 5 == 0:
                try:
                    from Information.comprehensive_diagnostics import log_server_logits
                    from Information.partial_collapse_diagnostics import (
                        check_server_head_capacity,
                        visualize_embeddings_pca
                    )
                    
                    # Check server head capacity
                    head_info, needs_capacity = check_server_head_capacity(net)
                    
                    # Log server logits
                    logits, probs, per_class_acc = log_server_logits(net, testloader, device, server_round)
                    
                    # Visualize embeddings (optional, requires sklearn)
                    if server_round == 0 or server_round % 10 == 0:
                        try:
                            import os
                            os.makedirs("visualizations", exist_ok=True)
                            vis_path = f"visualizations/embeddings_pca_round_{server_round}.png"
                            visualize_embeddings_pca(
                                [net], testloader, device,
                                args.num_classes if args.num_classes else 10,
                                server_round, vis_path
                            )
                        except Exception as e:
                            print(f"[Server] Embedding visualization skipped: {e}")
                    
                    # Save per-class accuracy to metrics
                    per_class_metrics = {f"class_{i}_acc": float(acc) for i, acc in enumerate(per_class_acc)}
                except Exception as e:
                    print(f"[Server] Debug logging failed: {e}")
                    import traceback
                    traceback.print_exc()
                    per_class_metrics = {}
            else:
                per_class_metrics = {}
            
            loss, accuracy = test(net, testloader, device)
            
            # Save aggregated model with per-class metrics
            metrics = {"loss": float(loss), "accuracy": float(accuracy), **per_class_metrics}
            db_round = server_round + args.first_round_offset
            contribution_db.save_aggregated_model(
                round_num=db_round,
                parameters=get_parameters(net),
                metrics=metrics
            )
            
            print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
            if per_class_metrics:
                num_classes = args.num_classes if args.num_classes else 10
                per_class_acc_list = [per_class_metrics.get(f"class_{i}_acc", 0.0) for i in range(num_classes)]
                print(f"  Per-class accuracy: {[f'{acc:.4f}' for acc in per_class_acc_list]}")
            
            return loss, {"accuracy": accuracy, **per_class_metrics}
        return evaluate
    
    def on_fit_config_fn(server_round):
        """Return configuration for fit phase."""
        return {"server_round": server_round}
    
    def aggregate_fit_metrics(metrics):
        """Aggregate metrics from multiple clients.
        Compatible with Flower's current format where each entry is
        (num_examples, metrics_dict). Falls back to dict-only entries.
        Computes weighted averages by num_examples when available.
        """
        aggregated = {}
        for metric_name in ["loss", "accuracy"]:
            total_weighted = 0.0
            total_weight = 0.0
            for entry in metrics:
                num_examples = 1
                metrics_dict = None
                if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], dict):
                    num_examples, metrics_dict = entry
                elif isinstance(entry, dict):
                    metrics_dict = entry
                if metrics_dict is None:
                    continue
                value = metrics_dict.get(metric_name)
                if value is None:
                    continue
                try:
                    v = float(value)
                except Exception:
                    continue
                try:
                    w = float(num_examples) if num_examples is not None else 1.0
                except Exception:
                    w = 1.0
                total_weighted += v * w
                total_weight += w
            if total_weight > 0:
                aggregated[metric_name] = total_weighted / total_weight
        return aggregated
    
    # Custom strategy to track contributions
    class ContributionTrackingStrategy(fl.server.strategy.FedAvg):
        def configure_fit(self, server_round, parameters, client_manager):
            """Configure fit to exclude certain clients."""
            # Get the default configuration from parent
            config = super().configure_fit(server_round, parameters, client_manager)
            
            # Log the in/out client list so excluded clients are visible. The
            # actual exclusion is enforced in aggregate_fit (post-hoc filter)
            # plus the lower min_fit_clients setting downstream. Flower 1.22
            # changed client_manager.all() to return a dict[str, ClientProxy]
            # rather than a list — handle both shapes safely.
            if excluded_client_ids:
                available = client_manager.all()
                if isinstance(available, dict):
                    available_proxies = list(available.values())
                else:
                    available_proxies = list(available)
                avail_cids = [getattr(c, "cid", str(c)) for c in available_proxies]
                # Convert excluded ids to strings to match Flower's cid type
                excluded_str = {str(x) for x in excluded_client_ids}
                kept = [cid for cid in avail_cids if cid not in excluded_str]
                print(
                    f"[Server] Round {server_round}: Available clients: {avail_cids}, "
                    f"Excluding: {sorted(excluded_client_ids)}, Filtered: {kept}"
                )
            
            return config
        
        def aggregate_fit(self, server_round, results, failures):
            """Aggregate fit results and track contributions."""
            # Debug: Comprehensive aggregation analysis (first round and every 5 rounds)
            if (server_round == 0 or server_round % 5 == 0) and results:
                try:
                    from Information.debug_training import analyze_aggregated_parameters
                    from Information.comprehensive_diagnostics import log_aggregation_details
                    
                    param_lists = []
                    client_ids = []
                    client_samples = []
                    
                    for _, fit_res in results:
                        # Check status
                        status_ok = True
                        try:
                            if hasattr(fit_res.status, 'is_success'):
                                status_ok = fit_res.status.is_success()
                            elif hasattr(fit_res.status, 'code'):
                                code = fit_res.status.code
                                if isinstance(code, int):
                                    status_ok = (code == 0)
                                elif isinstance(code, str):
                                    status_ok = (code.upper() == 'OK' or code.upper() == 'SUCCESS')
                                else:
                                    status_ok = fit_res.parameters is not None
                            else:
                                status_ok = fit_res.parameters is not None
                        except:
                            status_ok = fit_res.parameters is not None
                        
                        if status_ok:
                            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                            param_lists.append(params)
                            
                            # Extract client ID
                            client_id = fit_res.metrics.get("client_id", len(client_ids))
                            client_ids.append(client_id)
                            client_samples.append(fit_res.num_examples)
                    
                    if param_lists:
                        # Log aggregation details
                        log_aggregation_details(param_lists, client_samples, server_round)
                        
                        # Analyze parameters
                        # Get parameter names from model
                        param_names = list(net.state_dict().keys())
                        analyze_aggregated_parameters(param_lists, names=param_names, client_ids=client_ids)
                        
                        # Additional checks for partial collapse (classes 0 and 3 at 0% accuracy)
                        if server_round == 0:
                            try:
                                import numpy as np
                                from Information.partial_collapse_diagnostics import check_server_head_capacity
                                
                                # Check server head capacity
                                head_info, needs_capacity = check_server_head_capacity(net)
                                
                                # Check final layer weights for class bias
                                final_layer_idx = len(param_names) - 1
                                if final_layer_idx >= 0 and len(param_lists) > 0:
                                    final_layer_params = [p[final_layer_idx] for p in param_lists]
                                    if len(final_layer_params) > 0 and len(final_layer_params[0].shape) == 2:
                                        # This is a weight matrix (out_features x in_features)
                                        print(f"\n[Server] Final layer weight analysis (per-client, per-class):")
                                        for client_idx, (client_id, params) in enumerate(zip(client_ids, final_layer_params)):
                                            # Check if any class output has extreme weights
                                            for cls in range(params.shape[0]):
                                                cls_weights = params[cls, :]
                                                weight_magnitude = np.abs(cls_weights).mean()
                                                print(f"    Client {client_id}, Class {cls}: avg weight magnitude = {weight_magnitude:.6f}")
                            except Exception as e:
                                print(f"[Server] Partial collapse diagnostics failed: {e}")
                                import traceback
                                traceback.print_exc()
                except Exception as e:
                    print(f"[Server] Debug analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Filter out excluded clients from results before saving
            if excluded_client_ids:
                filtered_results = []
                for result_item in results:
                    # Extract client ID from result
                    client_id = None
                    try:
                        if isinstance(result_item, tuple) and len(result_item) == 2:
                            _, fit_res = result_item
                        else:
                            fit_res = result_item
                        
                        # Try to get client ID from metrics
                        if fit_res.metrics:
                            client_id = fit_res.metrics.get("client_id")
                        
                        # If client is excluded, skip it
                        if client_id is not None and client_id in excluded_client_ids:
                            print(f"[Server] Filtering out excluded client {client_id} from results")
                            continue
                    except Exception as e:
                        print(f"[Server] Error checking exclusion for result: {e}")
                    
                    filtered_results.append(result_item)
                results = filtered_results
            
            # Save each client's contribution before aggregation
            if results:
                print(f"[Server] Saving contributions for {len(results)} clients in round {server_round}")
                print(f"[Server] Results type: {type(results)}, First item type: {type(results[0]) if results else 'N/A'}")
                
                for idx, result_item in enumerate(results):
                    # Handle different Flower versions:
                    # - Older versions: results is [(client_proxy, fit_res), ...]
                    # - Newer versions (1.8+): results is [fit_res, ...] (list of FitRes objects)
                    try:
                        # Try to unpack as tuple (older Flower versions)
                        if isinstance(result_item, tuple) and len(result_item) == 2:
                            client_proxy, fit_res = result_item
                        else:
                            # Newer Flower versions: result_item is directly a FitRes object
                            fit_res = result_item
                            client_proxy = None
                    except (ValueError, TypeError) as e:
                        print(f"[Server] Error unpacking result {idx}: {e}, item type: {type(result_item)}")
                        continue
                    
                    # Check status - handle different Flower versions
                    status_ok = True
                    try:
                        # Try Flower 1.8+ style: status might have is_success() or code attribute
                        if hasattr(fit_res.status, 'is_success'):
                            status_ok = fit_res.status.is_success()
                        elif hasattr(fit_res.status, 'code'):
                            # Try to check code value (might be 0 for OK, or a string)
                            code = fit_res.status.code
                            if isinstance(code, int):
                                status_ok = (code == 0)  # 0 typically means OK
                            elif isinstance(code, str):
                                status_ok = (code.upper() == 'OK' or code.upper() == 'SUCCESS')
                            else:
                                # If we can't determine, assume OK if we have parameters
                                status_ok = fit_res.parameters is not None
                        else:
                            # If status doesn't have expected attributes, assume OK if parameters exist
                            status_ok = fit_res.parameters is not None
                    except Exception as e:
                        print(f"[Server] Error checking status for client {idx}: {e}, assuming OK if parameters exist")
                        # If status check fails, assume OK if we have parameters
                        status_ok = fit_res.parameters is not None
                    
                    if not status_ok:
                        print(f"[Server] Skipping client {idx}: status not OK")
                        continue
                    
                    client_id = None
                    
                    # Try to extract client ID from metrics (preferred method)
                    if fit_res.metrics:
                        client_id = fit_res.metrics.get("client_id")
                        if client_id is not None:
                            print(f"[Server] Client {idx}: extracted client_id={client_id} from metrics")
                    
                    # Extract from client proxy cid if available (only for older Flower versions)
                    if client_id is None and client_proxy is not None:
                        try:
                            client_id = int(client_proxy.cid)
                            print(f"[Server] Client {idx}: extracted client_id={client_id} from proxy.cid")
                        except (AttributeError, ValueError, TypeError) as e:
                            print(f"[Server] Client {idx}: could not extract from proxy.cid: {e}")
                    
                    # Fallback: use index as client ID
                    if client_id is None:
                        client_id = idx
                        print(f"[Server] Client {idx}: using index as client_id={client_id}")
                    
                    try:
                        parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
                        num_samples = fit_res.num_examples
                        db_round = server_round + args.first_round_offset
                        print(f"[Server] Saving contribution for client {client_id}: {num_samples} samples, {len(parameters)} parameter arrays (db_round={db_round})")

                        # Save contribution
                        contribution_db.save_contribution(
                            round_num=db_round,
                            client_id=client_id,
                            parameters=parameters,
                            num_samples=num_samples,
                            metrics=fit_res.metrics or {}
                        )
                        print(f"[Server] Successfully saved contribution for client {client_id}")
                    except Exception as e:
                        print(f"[Server] ERROR saving contribution for client {client_id}: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"[Server] WARNING: No results to save in round {server_round}")
            
            # Call parent aggregation
            return super().aggregate_fit(server_round, results, failures)
    
    # Define strategy (Federated Averaging with contribution tracking)
    # Calculate number of required clients (total minus excluded)
    total_clients = 4  # Total number of clients in the system
    num_required_clients = total_clients - len(excluded_client_ids)
    if num_required_clients < 1:
        raise ValueError(f"Cannot exclude all clients. Excluded: {excluded_client_ids}, Total: {total_clients}")
    
    strategy = ContributionTrackingStrategy(
        fraction_fit=1.0,  # Use 100% of available clients for training
        fraction_evaluate=1.0,  # Use 100% of available clients for evaluation
        min_fit_clients=num_required_clients,  # Require all non-excluded clients for training
        min_evaluate_clients=num_required_clients,  # Require all non-excluded clients for evaluation
        min_available_clients=num_required_clients,  # Require all non-excluded clients to be available
        evaluate_fn=get_evaluate_fn(),  # Server-side evaluation function
        on_fit_config_fn=on_fit_config_fn,  # Configuration function
        fit_metrics_aggregation_fn=aggregate_fit_metrics,  # Aggregate fit metrics
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(net)),
    )
    print(f"[Server] Strategy configured: requiring {num_required_clients} clients for each round (excluding {sorted(excluded_client_ids)})")
    
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

