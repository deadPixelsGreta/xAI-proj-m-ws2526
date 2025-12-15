"""Model builders and checkpoint helpers for supported architectures."""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


# Supported model architectures
SUPPORTED_MODELS = ["resnet18", "resnet34", "efficientnet_b0"]


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    device: torch.device = None,
) -> nn.Module:
    """Create a supported backbone and replace its classifier for num_classes.

    Raises ValueError if model_name is unsupported.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet34":
        if pretrained:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        if pretrained:
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Choose from {SUPPORTED_MODELS}"
        )

    if device is not None:
        model = model.to(device)

    return model


def load_checkpoint(
    checkpoint_path: str, device: torch.device, num_classes: int = 10
) -> Tuple[nn.Module, str, float]:
    """Load a checkpoint, rebuild the model, and return (model, model_name, val_acc)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model name from checkpoint or infer from filename
    if "model_name" in checkpoint:
        model_name = checkpoint["model_name"]
    else:
        # Try to infer from filename
        filename = Path(checkpoint_path).stem.lower()
        model_name = None
        for name in SUPPORTED_MODELS:
            if name in filename:
                model_name = name
                break
        if model_name is None:
            model_name = "resnet18"  # Default fallback

    # Get number of classes from checkpoint
    num_classes = checkpoint.get("num_classes", num_classes)

    # Create and load model (without pretrained weights)
    model = create_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_acc", 0.0)

    return model, model_name, val_acc
