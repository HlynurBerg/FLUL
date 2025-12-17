"""Flower client for federated learning."""
import argparse

import flwr as fl
import torch

from model import AVAILABLE_MODELS, create_model, get_parameters, set_parameters
from utils import load_data, train_epoch, test


def main():
    """Run Flower client."""
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        help="Client ID (0-indexed, determines which data partition to use)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="Total number of clients",
    )
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
        help="Model architecture to train",
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
        "--partition-type",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical", "class_vertical"],
        help="How to partition data: horizontal (by samples), vertical (by features), or class_vertical (by class)",
    )
    args = parser.parse_args()
    
    # Load model (adjust for different datasets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log device information
    if torch.cuda.is_available():
        print(f"[Client {args.client_id}] Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"[Client {args.client_id}] CUDA device count: {torch.cuda.device_count()}")
        print(f"[Client {args.client_id}] CUDA version: {torch.version.cuda}")
    else:
        print(f"[Client {args.client_id}] CUDA not available, using CPU")
    print(f"[Client {args.client_id}] Device: {device}")
    
    net = create_model(
        args.model,
        args.dataset,
        num_classes=args.num_classes,
        num_channels=args.num_channels,
        img_size=args.img_size,
    ).to(device)
    
    # Load data for this specific client
    # Use smaller batch size for large images (416x416) to avoid GPU memory issues
    batch_size = 8 if (args.img_size and args.img_size > 224) else 32
    print(f"[Client {args.client_id}] Using batch size: {batch_size} (adjusted for image size {args.img_size})")
    
    client_loaders, testloader = load_data(
        args.dataset,
        args.num_clients,
        batch_size=batch_size,
        partition_type=args.partition_type,
        dataset_path=args.dataset_path,
        img_size=args.img_size,
        num_channels=args.num_channels,
    )
    trainloader = client_loaders[args.client_id]
    
    print(f"Client {args.client_id}: Training samples: {len(trainloader.dataset)}, Test samples: {len(testloader.dataset)}")
    
    # Define Flower client
    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            """Return current model parameters."""
            return get_parameters(net)
        
        def fit(self, parameters, config):
            """Train the model on local data."""
            # Set model parameters
            set_parameters(net, parameters)
            
            # Get server round from config if available
            server_round = config.get("server_round", 0)
            
            print(f"[Client {args.client_id}] Round {server_round}: Starting training...")
            
            # Debug: Check class distribution (only on first round and first client)
            if server_round == 0 and args.client_id == 0:
                from debug_training import check_class_distribution
                check_class_distribution(trainloader, args.num_classes if args.num_classes else 10)
            
            # Train locally (enable debug on first round)
            # Use label smoothing for class_vertical to prevent overconfidence
            label_smoothing = 0.1 if args.partition_type == "class_vertical" else 0.0
            train_epoch(net, trainloader, device, epochs=1, debug=(server_round == 0), label_smoothing=label_smoothing)
            
            # Debug: Comprehensive logging (first round and every 5 rounds)
            if server_round == 0 or server_round % 5 == 0:
                from debug_training import log_model_outputs, log_gradient_norms
                from comprehensive_diagnostics import (
                    log_client_embeddings, log_gradient_norms_detailed,
                    check_batch_class_distribution, verify_no_activations_in_forward
                )
                from partial_collapse_diagnostics import (
                    check_embedding_variance_per_class,
                    check_per_batch_class_distribution as check_batch_dist
                )
                
                # Verify no softmax in forward
                sample_images, _ = next(iter(trainloader))
                sample_images = sample_images[:1].to(device)
                verify_no_activations_in_forward(net, sample_images)
                
                # Log embeddings and logits
                log_client_embeddings(net, trainloader, device, args.client_id, server_round)
                
                # Check embedding variance per class (critical for partial collapse)
                if args.partition_type == "class_vertical":
                    check_embedding_variance_per_class(
                        net, trainloader, device,
                        args.num_classes if args.num_classes else 10,
                        args.client_id, server_round
                    )
                
                # Log detailed gradients
                log_gradient_norms_detailed(net, server_round, args.client_id)
                
                # Check batch composition
                if server_round == 0:
                    check_batch_class_distribution(
                        trainloader, 
                        args.num_classes if args.num_classes else 10,
                        num_batches=5
                    )
                    # Also check per-batch distribution
                    check_batch_dist(
                        trainloader,
                        args.num_classes if args.num_classes else 10,
                        args.client_id,
                        num_batches=10
                    )
                
                # Log model outputs
                log_model_outputs(net, trainloader, device, f"Client {args.client_id} (after training)")
            
            print(f"[Client {args.client_id}] Round {server_round}: Starting evaluation...")
            # Evaluate local performance for metrics
            loss, accuracy = test(net, testloader, device)
            print(f"[Client {args.client_id}] Round {server_round}: Training complete - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Return updated parameters, number of training examples, and metrics
            metrics = {
                "client_id": args.client_id,
                "server_round": server_round,
                "local_loss": float(loss),
                "local_accuracy": float(accuracy),
                "num_training_samples": len(trainloader.dataset)
            }
            
            return get_parameters(net), len(trainloader.dataset), metrics
        
        def evaluate(self, parameters, config):
            """Evaluate the model on local test data."""
            set_parameters(net, parameters)
            loss, accuracy = test(net, testloader, device)
            return loss, len(testloader.dataset), {"accuracy": accuracy}
    
    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )


if __name__ == "__main__":
    main()

