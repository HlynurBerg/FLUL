"""Main entry point for running federated learning experiments."""
import argparse
import subprocess
import sys
import time

from model import AVAILABLE_MODELS


def run_server(
    dataset="MNIST",
    model_name="simplenet",
    *,
    dataset_path=None,
    img_size=None,
    num_channels=None,
    num_classes=None,
):
    """Run the Flower server."""
    print("Starting Flower server...")
    subprocess.run(
        [
            sys.executable,
            "server.py",
            "--dataset",
            dataset,
            "--model",
            model_name,
            *([] if dataset_path is None else ["--dataset-path", dataset_path]),
            *([] if img_size is None else ["--img-size", str(img_size)]),
            *([] if num_channels is None else ["--num-channels", str(num_channels)]),
            *([] if num_classes is None else ["--num-classes", str(num_classes)]),
        ]
    )


def run_client(
    client_id,
    num_clients=3,
    dataset="MNIST",
    model_name="simplenet",
    *,
    dataset_path=None,
    img_size=None,
    num_channels=None,
    num_classes=None,
    partition_type="horizontal",
):
    """Run a Flower client."""
    print(f"Starting Flower client {client_id}...")
    subprocess.run(
        [
            sys.executable,
            "client.py",
            "--client-id",
            str(client_id),
            "--num-clients",
            str(num_clients),
            "--dataset",
            dataset,
            "--model",
            model_name,
            "--partition-type",
            partition_type,
            *([] if dataset_path is None else ["--dataset-path", dataset_path]),
            *([] if img_size is None else ["--img-size", str(img_size)]),
            *([] if num_channels is None else ["--num-channels", str(num_channels)]),
            *([] if num_classes is None else ["--num-classes", str(num_classes)]),
        ]
    )


def main():
    """Main function to run federated learning setup."""
    parser = argparse.ArgumentParser(description="Federated Learning Testing Environment")
    parser.add_argument(
        "mode",
        choices=["server", "client", "demo"],
        help="Mode: 'server' to run server, 'client' to run client, 'demo' to run demo",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        help="Client ID (only used in client mode)",
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
        "--partition-type",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical"],
        help="Partition type when launching via app.py (client mode).",
    )
    
    args = parser.parse_args()
    
    if args.mode == "server":
        run_server(
            args.dataset,
            args.model,
            dataset_path=args.dataset_path,
            img_size=args.img_size,
            num_channels=args.num_channels,
            num_classes=args.num_classes,
        )
    elif args.mode == "client":
        run_client(
            args.client_id,
            args.num_clients,
            args.dataset,
            args.model,
            dataset_path=args.dataset_path,
            img_size=args.img_size,
            num_channels=args.num_channels,
            num_classes=args.num_classes,
            partition_type=args.partition_type,
        )
    elif args.mode == "demo":
        print("Running demo mode - starting server and 3 clients...")
        print("Note: In a real scenario, you would run these in separate terminals.")
        print("Starting server...")
        # For demo, we'll just provide instructions
        print("\nTo run the demo:")
        extra_flags = []
        if args.dataset_path:
            extra_flags.append(f"--dataset-path {args.dataset_path}")
        if args.img_size:
            extra_flags.append(f"--img-size {args.img_size}")
        if args.num_channels:
            extra_flags.append(f"--num-channels {args.num_channels}")
        if args.num_classes:
            extra_flags.append(f"--num-classes {args.num_classes}")
        extra = (" " + " ".join(extra_flags)) if extra_flags else ""
        print(
            "1. In one terminal: python app.py server --dataset {dataset} --model {model}{extra}".format(
                dataset=args.dataset,
                model=args.model,
                extra=extra,
            )
        )
        print("2. In separate terminals:")
        for i in range(args.num_clients):
            print(
                "   Terminal {term}: python app.py client --client-id {cid} --num-clients {total} "
                "--dataset {dataset} --model {model}{extra}".format(
                    term=i + 2,
                    cid=i,
                    total=args.num_clients,
                    dataset=args.dataset,
                    model=args.model,
                    extra=extra,
                )
            )


if __name__ == "__main__":
    main()

