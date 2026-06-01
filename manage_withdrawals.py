"""Tool to manage client withdrawals and recalculate models (git-like history management)."""
import argparse
from contributions_db import ContributionDB
from model import create_model, set_parameters, get_parameters
from utils import load_data, test
from unlearning import (
    GradientBasedUnlearning,
    InfluenceFunctionBasedUnlearning,
    HessianInfluenceUnlearning,
    ClassDiscriminativePruningUnlearning,
    GradientAscentKDUnlearning,
    evaluate_unlearned_model,
)
import torch


def parse_sample_ids(s: str):
    """Comma-separated global sample ids."""
    if not s or not str(s).strip():
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def withdraw_samples(db, round_num, client_id, sample_ids, reason=None, contribution_key=None):
    """Withdraw specific samples from one contribution (latest for round/client unless key given)."""
    ck = db.resolve_contribution_key(round_num, client_id, contribution_key)
    return db.withdraw_samples_from_contribution(ck, sample_ids, reason=reason)


def restore_samples(db, round_num, client_id, sample_ids=None, all_samples=False, contribution_key=None):
    """Restore specific or all sample-level withdrawals for a contribution."""
    ck = db.resolve_contribution_key(round_num, client_id, contribution_key)
    return db.restore_samples_from_contribution(ck, sample_ids=sample_ids, all_samples=all_samples)


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
        ws = entry.get("withdrawn_sample_ids") or []
        if ws:
            print(f"    Partial sample withdrawal: {len(ws)} id(s) — {ws[:20]}{'...' if len(ws) > 20 else ''}")
        
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
        print("No clients are currently withdrawn (full-client withdrawal).")
    else:
        for client_id in withdrawn:
            info = db.withdrawals["withdrawn_clients"][client_id]
            print(f"\nClient {client_id}:")
            print(f"  Withdrawn At: {info['withdrawn_at']}")
            print(f"  Reason: {info.get('reason', 'Not specified')}")
            print(f"  Affects Rounds: {info.get('affects_rounds', [])}")
    
    db._ensure_sample_withdrawals_schema()
    sw = db.withdrawals.get("sample_withdrawals", {}).get("active", {})
    print(f"\n--- Sample-level withdrawals ({len(sw)} contribution(s)) ---")
    if not sw:
        print("  (none)")
    else:
        for ck, rec in sw.items():
            ids = rec.get("sample_ids", [])
            print(f"  {ck}")
            print(f"    withdrawn sample_ids ({len(ids)}): {ids[:24]}{'...' if len(ids) > 24 else ''}")
            print(f"    at: {rec.get('withdrawn_at', '')}")
    
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
                   dataset_path=None, num_clients=3, **kwargs):
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

    # Handle dataset name (strip _CLASS_VERTICAL suffix if present)
    base_dataset = dataset_name.split("_")[0] if "_" in dataset_name else dataset_name

    print(f"\n{'='*60}")
    if algorithm_name == "gradient":
        print(f"UNLEARNING CLIENT {client_id} (Gradient-Based Algorithm)")
        unlearner = GradientBasedUnlearning(db)
    elif algorithm_name == "influence":
        print(f"UNLEARNING CLIENT {client_id} (Influence Function-Based Algorithm)")
        damping = kwargs.get("damping_factor", 0.01)
        influence_scale = kwargs.get("influence_scale", 1.0)
        unlearner = InfluenceFunctionBasedUnlearning(db, damping_factor=damping)
    elif algorithm_name == "hessian":
        print(f"UNLEARNING CLIENT {client_id} (Hessian IHVP Algorithm)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if base_dataset == "CUSTOM":
            if num_classes is None or num_channels is None or img_size is None:
                print("Error: --num-classes, --num-channels, and --img-size are required for CUSTOM datasets")
                return None
            net = create_model(model_name, base_dataset,
                               num_classes=num_classes, num_channels=num_channels, img_size=img_size).to(device)
        else:
            net = create_model(model_name, base_dataset).to(device)
        influence_scale = kwargs.get("influence_scale", 1.0)
        unlearner = HessianInfluenceUnlearning(
            db,
            net,
            dataset_name,
            num_clients=num_clients,
            device=device,
            damping=kwargs.get("damping_factor", 0.1),
            cg_max_iter=kwargs.get("cg_max_iter", 50),
            cg_tol=kwargs.get("cg_tol", 1e-4),
        )
    elif algorithm_name == "class_pruning":
        print(f"UNLEARNING CLIENT {client_id} (Class-Discriminative Pruning, Wang et al. 2022)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if base_dataset == "CUSTOM":
            if num_classes is None or num_channels is None or img_size is None:
                print("Error: --num-classes, --num-channels, and --img-size are required for CUSTOM datasets")
                return None
            net = create_model(model_name, base_dataset,
                               num_classes=num_classes, num_channels=num_channels, img_size=img_size).to(device)
        else:
            net = create_model(model_name, base_dataset).to(device)
        influence_scale = kwargs.get("influence_scale", 1.0)
        unlearner = ClassDiscriminativePruningUnlearning(
            db,
            net,
            dataset_name,
            num_clients=num_clients,
            device=device,
            prune_ratio=kwargs.get("prune_ratio", 0.1),
            n_probe_per_class=kwargs.get("n_probe_per_class", 64),
            finetune_epochs=kwargs.get("finetune_epochs", 0),
            finetune_lr=kwargs.get("finetune_lr", 1e-3),
        )
    elif algorithm_name == "gradient_ascent_kd":
        print(f"UNLEARNING CLIENT {client_id} (Gradient Ascent + KD, SCRUB-style)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if base_dataset == "CUSTOM":
            if num_classes is None or num_channels is None or img_size is None:
                print("Error: --num-classes, --num-channels, and --img-size are required for CUSTOM datasets")
                return None
            net = create_model(model_name, base_dataset,
                               num_classes=num_classes, num_channels=num_channels, img_size=img_size).to(device)
        else:
            net = create_model(model_name, base_dataset).to(device)
        influence_scale = kwargs.get("influence_scale", 1.0)
        unlearner = GradientAscentKDUnlearning(
            db,
            net,
            dataset_name,
            num_clients=num_clients,
            device=device,
            epochs=kwargs.get("ga_epochs", 3),
            lr=kwargs.get("finetune_lr", 1e-3),
            alpha_retain=kwargs.get("alpha_retain", 1.0),
            gamma_kd=kwargs.get("gamma_kd", 1.0),
            beta_forget=kwargs.get("beta_forget", 1.0),
            temperature=kwargs.get("kd_temperature", 4.0),
            max_forget_steps=kwargs.get("max_forget_steps", None),
        )
    else:
        print(
            f"Error: Unknown algorithm '{algorithm}'. Use 'gradient', 'influence', "
            f"'hessian', 'class_pruning', or 'gradient_ascent_kd'"
        )
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
    
    if algorithm_name in ("influence", "hessian", "class_pruning", "gradient_ascent_kd"):
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


def unlearn_samples(
    db,
    dataset_name,
    round_num,
    client_id,
    sample_ids=None,
    contribution_key=None,
    influence_scale=1.0,
    damping_factor=0.01,
    evaluate=True,
    model_name="simplenet",
    num_classes=None,
    num_channels=None,
    img_size=None,
    dataset_path=None,
    algorithm="influence",
    num_clients=3,
    cg_max_iter=50,
    cg_tol=1e-4,
    prune_ratio=0.1,
    n_probe_per_class=64,
    finetune_epochs=0,
    finetune_lr=1e-3,
    ga_epochs=3,
    alpha_retain=1.0,
    gamma_kd=1.0,
    beta_forget=1.0,
    kd_temperature=4.0,
    max_forget_steps=None,
):
    """
    Unlearn specific samples from one client's contribution using per-sample
    influence estimation (Eq. 3.2–3.3 adapted to sample granularity).

    If ``sample_ids`` is None, the withdrawn sample IDs are read from the
    database (i.e. whatever was marked via ``withdraw-samples``).
    """
    algorithm_name = algorithm.lower()
    base_dataset = dataset_name.split("_")[0] if "_" in dataset_name else dataset_name

    print(f"\n{'='*60}")
    print(f"SAMPLE-LEVEL UNLEARNING  (round {round_num}, client {client_id}, algorithm={algorithm_name})")
    print(f"{'='*60}")

    def _build_torch_model_or_none():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if base_dataset == "CUSTOM":
            if num_classes is None or num_channels is None or img_size is None:
                print("Error: --num-classes, --num-channels, and --img-size are required for CUSTOM datasets")
                return None, None
            return create_model(model_name, base_dataset,
                                num_classes=num_classes, num_channels=num_channels, img_size=img_size).to(device), device
        return create_model(model_name, base_dataset).to(device), device

    def _resolve_sample_ids():
        if sample_ids:
            return sample_ids
        ck = db.resolve_contribution_key(round_num, client_id, contribution_key)
        state = db.get_sample_withdrawal_state(ck)
        if not state:
            print(f"Error: No active sample withdrawals for contribution {ck}")
            return None
        return [int(s) for s in state["sample_ids"]]

    if algorithm_name == "hessian":
        net, device = _build_torch_model_or_none()
        if net is None:
            return None
        unlearner = HessianInfluenceUnlearning(
            db,
            net,
            dataset_name,
            num_clients=num_clients,
            device=device,
            damping=damping_factor,
            cg_max_iter=cg_max_iter,
            cg_tol=cg_tol,
        )
        resolved_ids = _resolve_sample_ids()
        if resolved_ids is None:
            return None
        unlearned_params, metadata = unlearner.unlearn_samples_from_contribution(
            round_num=round_num,
            client_id=client_id,
            sample_ids=resolved_ids,
            influence_scale=influence_scale,
        )
        print(f"  Samples unlearned : {len(metadata['sample_ids'])}")
        print(f"  FL scale          : {metadata['fl_scale']:.6f}")
        print(f"  Client weight     : {metadata['client_weight']} / {metadata['total_weight']}")
        save_metrics = {
            "unlearned_samples": True,
            "unlearned_client": client_id,
            "unlearned_sample_ids": metadata["sample_ids"],
            "method": metadata["method"],
        }
    elif algorithm_name == "class_pruning":
        net, device = _build_torch_model_or_none()
        if net is None:
            return None
        unlearner = ClassDiscriminativePruningUnlearning(
            db,
            net,
            dataset_name,
            num_clients=num_clients,
            device=device,
            prune_ratio=prune_ratio,
            n_probe_per_class=n_probe_per_class,
            finetune_epochs=finetune_epochs,
            finetune_lr=finetune_lr,
        )
        resolved_ids = _resolve_sample_ids()
        if resolved_ids is None:
            return None
        unlearned_params, metadata = unlearner.unlearn_samples_from_contribution(
            round_num=round_num,
            client_id=client_id,
            sample_ids=resolved_ids,
            influence_scale=influence_scale,
        )
        print(f"  Samples unlearned : {len(metadata['sample_ids'])}")
        print(f"  Target classes    : {metadata['target_classes']}")
        print(f"  Prune ratio       : {metadata['prune_ratio']}")
        print(f"  Channels pruned   : {metadata['pruned_channels_per_layer']}")
        save_metrics = {
            "unlearned_samples": True,
            "unlearned_client": client_id,
            "unlearned_sample_ids": metadata["sample_ids"],
            "method": metadata["method"],
            "prune_ratio": metadata["prune_ratio"],
            "target_classes": metadata["target_classes"],
        }
    elif algorithm_name == "gradient_ascent_kd":
        net, device = _build_torch_model_or_none()
        if net is None:
            return None
        unlearner = GradientAscentKDUnlearning(
            db,
            net,
            dataset_name,
            num_clients=num_clients,
            device=device,
            epochs=ga_epochs,
            lr=finetune_lr,
            alpha_retain=alpha_retain,
            gamma_kd=gamma_kd,
            beta_forget=beta_forget,
            temperature=kd_temperature,
            max_forget_steps=max_forget_steps,
        )
        resolved_ids = _resolve_sample_ids()
        if resolved_ids is None:
            return None
        unlearned_params, metadata = unlearner.unlearn_samples_from_contribution(
            round_num=round_num,
            client_id=client_id,
            sample_ids=resolved_ids,
            influence_scale=influence_scale,
        )
        print(f"  Samples unlearned : {len(metadata['sample_ids'])}")
        print(f"  Epochs / LR       : {metadata['epochs']} / {metadata['lr']}")
        print(f"  Steps (forget/retain): {metadata['forget_steps_taken']} / {metadata['retain_steps_taken']}")
        save_metrics = {
            "unlearned_samples": True,
            "unlearned_client": client_id,
            "unlearned_sample_ids": metadata["sample_ids"],
            "method": metadata["method"],
            "epochs": metadata["epochs"],
        }
    else:
        # Default: influence-function-based (first-order)
        unlearner = InfluenceFunctionBasedUnlearning(db, damping_factor=damping_factor)
        if sample_ids:
            unlearned_params, metadata = unlearner.unlearn_samples_from_contribution(
                round_num=round_num,
                client_id=client_id,
                sample_ids=sample_ids,
                influence_scale=influence_scale,
            )
        else:
            unlearned_params, metadata = unlearner.unlearn_withdrawn_samples(
                round_num=round_num,
                client_id=client_id,
                influence_scale=influence_scale,
                contribution_key=contribution_key,
            )
        print(f"  Samples unlearned : {len(metadata['sample_ids'])}")
        print(f"  Weight fraction   : {metadata['sample_weight_fraction']:.6f}")
        print(f"  Weight source     : {metadata['weight_source']}")
        print(f"  Client weight     : {metadata['client_weight']} / {metadata['total_weight']}")
        save_metrics = {
            "unlearned_samples": True,
            "unlearned_client": client_id,
            "unlearned_sample_ids": metadata["sample_ids"],
            "method": metadata["method"],
            "weight_source": metadata["weight_source"],
        }

    db.save_aggregated_model(
        round_num=round_num,
        parameters=unlearned_params,
        metrics=save_metrics,
    )

    if evaluate:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_dataset = dataset_name.split("_")[0] if "_" in dataset_name else dataset_name
        if base_dataset == "CUSTOM":
            if num_classes is None or num_channels is None or img_size is None:
                print("  Warning: skipping evaluation — missing dataset parameters")
                evaluate = False
        if evaluate:
            from model import create_model, set_parameters
            from utils import load_data, test
            net = create_model(
                model_name, base_dataset,
                **(dict(num_classes=num_classes, num_channels=num_channels, img_size=img_size)
                   if base_dataset == "CUSTOM" else {})
            ).to(device)
            _, testloader = load_data(
                base_dataset,
                num_clients=1,
                batch_size=32,
                dataset_path=dataset_path,
                img_size=img_size,
                num_channels=num_channels,
            )
            set_parameters(net, unlearned_params)
            from utils import test
            loss, accuracy = test(net, testloader, device)
            print(f"  Evaluation        : Loss={loss:.4f}, Accuracy={accuracy:.4f}")

    print(f"{'='*60}\n")
    return metadata


def unlearn_samples_all_rounds(
    db,
    dataset_name,
    client_id,
    sample_ids=None,
    algorithm="influence",
    propagate=True,
    influence_scale=1.0,
    damping_factor=0.01,
    model_name="simplenet",
    num_classes=None,
    num_channels=None,
    img_size=None,
    dataset_path=None,
    num_clients=3,
    cg_max_iter=50,
    cg_tol=1e-4,
    prune_ratio=0.1,
    n_probe_per_class=64,
    finetune_epochs=0,
    finetune_lr=1e-3,
    ga_epochs=3,
    alpha_retain=1.0,
    gamma_kd=1.0,
    beta_forget=1.0,
    kd_temperature=4.0,
    max_forget_steps=None,
):
    """
    Unlearn specific samples from a client across all rounds they appear in.

    If ``sample_ids`` is None, the union of all active sample withdrawals for
    this client is read from the database.

    Args:
        db: ContributionDB instance.
        dataset_name: Dataset name.
        client_id: Client whose samples should be unlearned.
        sample_ids: Global sample IDs to remove.  None → read from DB.
        algorithm: ``"influence"`` (first-order, default) or ``"hessian"``.
        propagate: If True, recalculate subsequent rounds after unlearning.
        influence_scale: Scaling factor for the influence/IHVP correction.
        damping_factor: Damping for influence unlearning (clamping strength).
        model_name / num_classes / num_channels / img_size / dataset_path:
            Model/dataset arguments forwarded to HessianInfluenceUnlearning.
        num_clients: Number of FL clients (needed to reconstruct partitions for Hessian).
        cg_max_iter / cg_tol: Conjugate-gradient settings for Hessian method.

    Returns:
        Result dict from ``unlearn_samples_all_rounds()`` on the chosen unlearner.
    """
    algorithm_name = algorithm.lower()
    base_dataset = dataset_name.split("_")[0] if "_" in dataset_name else dataset_name

    # Resolve sample_ids from active withdrawals if not provided
    if not sample_ids:
        all_contributions = db.list_contributions(exclude_withdrawn=False)
        client_contributions = [c for c in all_contributions if c["client_id"] == client_id]
        merged_ids = set()
        for contrib in client_contributions:
            ck = contrib.get("contribution_key")
            if ck:
                state = db.get_sample_withdrawal_state(ck)
                if state:
                    merged_ids.update(int(s) for s in state.get("sample_ids", []))
        if not merged_ids:
            print(f"[manage_withdrawals] No active sample withdrawals for client {client_id}")
            return {"error": f"No active sample withdrawals for client {client_id}"}
        sample_ids = sorted(merged_ids)

    print(f"\n{'='*60}")
    print(f"MULTI-ROUND SAMPLE UNLEARNING  (client {client_id}, algorithm={algorithm_name})")
    print(f"  Sample IDs : {sample_ids}")
    print(f"  Propagate  : {propagate}")
    print(f"{'='*60}")

    def _build_torch_model_or_none():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if base_dataset == "CUSTOM":
            if num_classes is None or num_channels is None or img_size is None:
                print("Error: --num-classes, --num-channels, and --img-size are required for CUSTOM datasets")
                return None, None
            return create_model(model_name, base_dataset,
                                num_classes=num_classes, num_channels=num_channels, img_size=img_size).to(device), device
        return create_model(model_name, base_dataset).to(device), device

    if algorithm_name == "hessian":
        net, device = _build_torch_model_or_none()
        if net is None:
            return None
        unlearner = HessianInfluenceUnlearning(
            db,
            net,
            dataset_name,
            num_clients=num_clients,
            device=device,
            damping=damping_factor,
            cg_max_iter=cg_max_iter,
            cg_tol=cg_tol,
        )
    elif algorithm_name == "class_pruning":
        net, device = _build_torch_model_or_none()
        if net is None:
            return None
        unlearner = ClassDiscriminativePruningUnlearning(
            db,
            net,
            dataset_name,
            num_clients=num_clients,
            device=device,
            prune_ratio=prune_ratio,
            n_probe_per_class=n_probe_per_class,
            finetune_epochs=finetune_epochs,
            finetune_lr=finetune_lr,
        )
    elif algorithm_name == "gradient_ascent_kd":
        net, device = _build_torch_model_or_none()
        if net is None:
            return None
        unlearner = GradientAscentKDUnlearning(
            db,
            net,
            dataset_name,
            num_clients=num_clients,
            device=device,
            epochs=ga_epochs,
            lr=finetune_lr,
            alpha_retain=alpha_retain,
            gamma_kd=gamma_kd,
            beta_forget=beta_forget,
            temperature=kd_temperature,
            max_forget_steps=max_forget_steps,
        )
    elif algorithm_name == "gradient":
        print(
            "Error: 'gradient' algorithm only supports client-level unlearning "
            "(no sample-level support). Use 'influence', 'hessian', 'class_pruning', "
            "or 'gradient_ascent_kd' for sample-level unlearning."
        )
        return None
    else:
        unlearner = InfluenceFunctionBasedUnlearning(db, damping_factor=damping_factor)

    results = unlearner.unlearn_samples_all_rounds(
        client_id=client_id,
        sample_ids=sample_ids,
        dataset_name=dataset_name,
        propagate=propagate,
        influence_scale=influence_scale,
    )

    successful = [r for r in results.get("unlearned_rounds", []) if r["status"] == "success"]
    failed = [r for r in results.get("unlearned_rounds", []) if r["status"] != "success"]
    print(f"\n  Rounds unlearned : {[r['round'] for r in successful]}")
    if failed:
        print(f"  Rounds failed    : {[(r['round'], r.get('error')) for r in failed]}")
    print(f"  Propagated       : {results.get('propagated')}")
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
    
    # Load original models for comparison.
    #
    # Convention (see CODEBASE_MAP.md §3 / pitfall #4): the unlearning module
    # never adds an "unlearned" filename suffix, so filtering by the substring
    # is a no-op. The original FL aggregate is the OLDEST aggregate file in
    # each round dir (by mtime); newer aggregates were written by subsequent
    # unlearning runs. Use mtime ordering instead of string filtering.
    original_models = {}
    for round_num in affected_rounds:
        round_dir = db.contributions_dir / f"round_{round_num:04d}"
        aggregated_files = list(round_dir.glob("round_*_aggregated_*_params.pkl"))
        if aggregated_files:
            oldest_original = min(aggregated_files, key=lambda p: p.stat().st_mtime)
            with open(oldest_original, 'rb') as f:
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
        choices=[
            "withdraw",
            "restore",
            "withdraw-samples",
            "restore-samples",
            "history",
            "list",
            "recalculate",
            "unlearn",
            "unlearn-samples",
            "compare",
        ],
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
        "--sample-ids",
        type=str,
        default=None,
        help="Comma-separated global sample ids (for withdraw-samples / restore-samples)",
    )
    parser.add_argument(
        "--contribution-key",
        type=str,
        default=None,
        help="Exact contribution_key if not using latest for round+client",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Restore all sample withdrawals for that contribution (restore-samples)",
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
        choices=["gradient", "influence", "hessian", "class_pruning", "gradient_ascent_kd"],
        help=(
            "Unlearning algorithm to use (for unlearn/unlearn-samples command). "
            "'gradient' is client-level only; the others support sample-level."
        ),
    )
    parser.add_argument(
        "--damping-factor",
        type=float,
        default=0.01,
        help="Damping factor for influence-based algorithm or Hessian regularisation (default 0.01; hessian default 0.1)",
    )
    parser.add_argument(
        "--influence-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the computed correction (default 1.0)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="Number of clients used during training; needed by --algorithm hessian to reconstruct the data partition",
    )
    parser.add_argument(
        "--cg-max-iter",
        type=int,
        default=50,
        help="Max conjugate gradient iterations for --algorithm hessian (default 50)",
    )
    parser.add_argument(
        "--cg-tol",
        type=float,
        default=1e-4,
        help="CG convergence tolerance for --algorithm hessian (default 1e-4)",
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.1,
        help="(class_pruning) fraction of channels to prune per conv layer per target class (default 0.1)",
    )
    parser.add_argument(
        "--n-probe-per-class",
        type=int,
        default=64,
        help="(class_pruning) probe samples per class for TF-IDF computation (default 64)",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=0,
        help="(class_pruning) post-prune fine-tune epochs on retain data (default 0 = no fine-tune)",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-3,
        help="(class_pruning / gradient_ascent_kd) fine-tune learning rate (default 1e-3)",
    )
    parser.add_argument(
        "--ga-epochs",
        type=int,
        default=3,
        help="(gradient_ascent_kd) outer epochs for gradient ascent + KD fine-tuning (default 3)",
    )
    parser.add_argument(
        "--alpha-retain",
        type=float,
        default=1.0,
        help="(gradient_ascent_kd) weight on retain CE loss (default 1.0)",
    )
    parser.add_argument(
        "--gamma-kd",
        type=float,
        default=1.0,
        help="(gradient_ascent_kd) weight on KD loss (default 1.0)",
    )
    parser.add_argument(
        "--beta-forget",
        type=float,
        default=1.0,
        help="(gradient_ascent_kd) weight on forget ascent loss (default 1.0)",
    )
    parser.add_argument(
        "--kd-temperature",
        type=float,
        default=4.0,
        help="(gradient_ascent_kd) KD softmax temperature (default 4.0)",
    )
    parser.add_argument(
        "--max-forget-steps",
        type=int,
        default=None,
        help="(gradient_ascent_kd) cap forget batches per epoch (default unlimited)",
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
    
    elif args.command == "withdraw-samples":
        if args.client_id is None or args.round is None:
            print("Error: withdraw-samples requires --client-id and --round")
            return
        ids = parse_sample_ids(args.sample_ids)
        if not ids:
            print("Error: withdraw-samples requires --sample-ids (comma-separated)")
            return
        withdraw_samples(
            db,
            args.round,
            args.client_id,
            ids,
            reason=args.reason,
            contribution_key=args.contribution_key,
        )
    
    elif args.command == "restore-samples":
        if args.client_id is None or args.round is None:
            print("Error: restore-samples requires --client-id and --round")
            return
        if args.all_samples:
            restore_samples(
                db,
                args.round,
                args.client_id,
                all_samples=True,
                contribution_key=args.contribution_key,
            )
        else:
            ids = parse_sample_ids(args.sample_ids)
            if not ids:
                print("Error: pass --sample-ids or --all-samples")
                return
            restore_samples(
                db,
                args.round,
                args.client_id,
                sample_ids=ids,
                contribution_key=args.contribution_key,
            )
    
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
            num_clients=args.num_clients,
            damping_factor=args.damping_factor,
            influence_scale=args.influence_scale,
            cg_max_iter=args.cg_max_iter,
            cg_tol=args.cg_tol,
            prune_ratio=args.prune_ratio,
            n_probe_per_class=args.n_probe_per_class,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            ga_epochs=args.ga_epochs,
            alpha_retain=args.alpha_retain,
            gamma_kd=args.gamma_kd,
            beta_forget=args.beta_forget,
            kd_temperature=args.kd_temperature,
            max_forget_steps=args.max_forget_steps,
        )

    elif args.command == "unlearn-samples":
        if args.client_id is None or args.round is None:
            print("Error: unlearn-samples requires --client-id and --round")
            return
        ids = parse_sample_ids(args.sample_ids) if args.sample_ids else None
        unlearn_samples(
            db,
            args.dataset,
            args.round,
            args.client_id,
            sample_ids=ids,
            contribution_key=args.contribution_key,
            influence_scale=args.influence_scale,
            damping_factor=args.damping_factor,
            evaluate=not args.no_evaluate,
            model_name=args.model,
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            img_size=args.img_size,
            dataset_path=args.dataset_path,
            algorithm=args.algorithm,
            num_clients=args.num_clients,
            cg_max_iter=args.cg_max_iter,
            cg_tol=args.cg_tol,
            prune_ratio=args.prune_ratio,
            n_probe_per_class=args.n_probe_per_class,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            ga_epochs=args.ga_epochs,
            alpha_retain=args.alpha_retain,
            gamma_kd=args.gamma_kd,
            beta_forget=args.beta_forget,
            kd_temperature=args.kd_temperature,
            max_forget_steps=args.max_forget_steps,
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

