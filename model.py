"""Model definitions and helpers for federated learning."""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18

AVAILABLE_MODELS = ("simplenet", "resnet18")

DEFAULT_DATASET_CONFIGS: Dict[str, Dict[str, int]] = {
    "CIFAR10": {"num_classes": 10, "num_channels": 3, "img_size": 32},
    "MNIST": {"num_classes": 10, "num_channels": 1, "img_size": 28},
    "FASHIONMNIST": {"num_classes": 10, "num_channels": 1, "img_size": 28},
}


def _dataset_config(
    dataset_name: str,
    overrides: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, int]:
    """Return canonical channel/count settings for supported datasets."""
    config = DEFAULT_DATASET_CONFIGS.get(dataset_name.upper(), {}).copy()
    overrides = overrides or {}
    for key in ("num_classes", "num_channels", "img_size"):
        value = overrides.get(key)
        if value is not None:
            config[key] = value
    missing = [k for k in ("num_classes", "num_channels", "img_size") if k not in config]
    if missing:
        raise ValueError(
            f"Missing dataset configuration values: {missing}. "
            "Specify --num-classes, --num-channels, and --img-size when using custom datasets."
        )
    return config


class SimpleNet(nn.Module):
    """Lightweight CNN for baseline federated learning experiments."""

    def __init__(self, num_classes: int = 10, num_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.img_size = img_size
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        pool_out_size = (img_size // 4) ** 2
        self.fc1 = nn.Linear(64 * pool_out_size, 128)
        # Add BatchNorm before final layer to stabilize features
        self.bn = nn.BatchNorm1d(128)
        # Use 2-layer MLP for better capacity (helps prevent partial class collapse)
        self.fc2_hidden = nn.Linear(128, 64, bias=False)
        self.fc2 = nn.Linear(64, num_classes, bias=False)
        self.relu_fc = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        pool_out_size = (self.img_size // 4) ** 2
        x = x.view(-1, 64 * pool_out_size)
        x = self.relu(self.fc1(x))
        x = self.bn(x)  # Normalize before final layer
        x = self.dropout(x)
        # 2-layer MLP for better capacity
        x = self.relu_fc(self.fc2_hidden(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)  # Raw logits - no softmax
        return x


def _build_resnet18(num_classes: int, num_channels: int, img_size: int) -> nn.Module:
    """Create a ResNet-18 variant supporting both small and large images."""
    net = resnet18(weights=None)

    if img_size <= 64:
        # CIFAR-style stem
        net.conv1 = nn.Conv2d(
            num_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        net.maxpool = nn.Identity()
    else:
        # Use ImageNet stem but adapt channels
        net.conv1 = nn.Conv2d(
            num_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

    # Replace final layer with a 2-layer MLP for better capacity
    # This helps prevent partial class collapse
    fc_in_features = net.fc.in_features
    hidden_dim = max(128, fc_in_features // 2)  # Hidden layer with reasonable size
    
    net.fc = nn.Sequential(
        nn.LayerNorm(fc_in_features),
        nn.Linear(fc_in_features, hidden_dim, bias=False),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, num_classes, bias=False)  # No bias to prevent class bias
    )
    return net


def create_model(
    model_name: str,
    dataset_name: str,
    *,
    num_classes: Optional[int] = None,
    num_channels: Optional[int] = None,
    img_size: Optional[int] = None,
) -> nn.Module:
    """Factory that instantiates the requested architecture for a dataset."""
    overrides = {
        "num_classes": num_classes,
        "num_channels": num_channels,
        "img_size": img_size,
    }
    config = _dataset_config(dataset_name, overrides)
    architecture = model_name.lower()

    if architecture == "simplenet":
        return SimpleNet(**config)
    if architecture == "resnet18":
        return _build_resnet18(config["num_classes"], config["num_channels"], config["img_size"])

    raise ValueError(
        f"Unknown model '{model_name}'. Available options: {', '.join(AVAILABLE_MODELS)}"
    )


def get_parameters(net):
    """Get model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    """Set model parameters from a list of numpy arrays."""
    # Get the device of the model (assumes all parameters are on the same device)
    device = next(net.parameters()).device
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)

