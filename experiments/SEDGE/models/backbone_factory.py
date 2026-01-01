"""Factory functions to create frozen pretrained backbones for SEDGE."""

from pathlib import Path
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torchvision import models

from experiments.SEDGE.training.console_ui import Colors, color


# Backbone configurations: (name, model_fn, weights, get_num_features_fn)
BACKBONE_CONFIGS = {
    "resnet34": {
        "model_fn": models.resnet34,
        "weights": models.ResNet34_Weights.IMAGENET1K_V1,
        "num_features": 512,  # Output of avgpool before fc
        "classifier_attr": "fc",
    },
    "densenet121": {
        "model_fn": models.densenet121,
        "weights": models.DenseNet121_Weights.IMAGENET1K_V1,
        "num_features": 1024,
        "classifier_attr": "classifier",
    },
    "efficientnet_b0": {
        "model_fn": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "num_features": 1280,
        "classifier_attr": "classifier",  # Sequential with Linear at index 1
    },
    "vit_b_16": {
        "model_fn": models.vit_b_16,
        "weights": models.ViT_B_16_Weights.IMAGENET1K_V1,
        "num_features": 768,
        "classifier_attr": "heads",  # Sequential with Linear at index 0
    },
}


def create_frozen_backbone(
    name: str,
    num_classes: int,
    device: torch.device = None,
) -> Tuple[nn.Module, int]:
    """
    Create a frozen pretrained backbone with its classifier replaced.

    Returns:
        Tuple of (model, feature_dim) where feature_dim is the input size
        to the original classifier (useful for adapters).
    """
    name = name.lower()
    if name not in BACKBONE_CONFIGS:
        raise ValueError(
            f"Unknown backbone: {name}. Available: {list(BACKBONE_CONFIGS.keys())}"
        )

    config = BACKBONE_CONFIGS[name]

    # Load pretrained model
    model = config["model_fn"](weights=config["weights"])

    # Replace classifier head with identity or simple projection to num_classes
    # For SEDGE, we keep the original classifier since adapters sit on top
    # But we need to ensure output is num_classes
    if name == "resnet34":
        model.fc = nn.Linear(config["num_features"], num_classes)
    elif name == "densenet121":
        model.classifier = nn.Linear(config["num_features"], num_classes)
    elif name == "efficientnet_b0":
        model.classifier[1] = nn.Linear(config["num_features"], num_classes)
    elif name == "vit_b_16":
        model.heads[0] = nn.Linear(config["num_features"], num_classes)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Move to device
    if device is not None:
        model = model.to(device)

    model.eval()  # Set to eval mode since we're not training it

    return model, config["num_features"]


def load_frozen_backbone_from_checkpoint(
    name: str,
    checkpoint_path: str,
    num_classes: int,
    device: torch.device = None,
) -> Tuple[nn.Module, int]:
    """
    Load a frozen backbone from a fine-tuned checkpoint file.

    Args:
        name: Backbone name (e.g., 'resnet34')
        checkpoint_path: Path to the .pth checkpoint file
        num_classes: Number of output classes
        device: Device to load model onto

    Returns:
        Tuple of (model, feature_dim)
    """
    name = name.lower()
    if name not in BACKBONE_CONFIGS:
        raise ValueError(
            f"Unknown backbone: {name}. Available: {list(BACKBONE_CONFIGS.keys())}"
        )

    config = BACKBONE_CONFIGS[name]

    # Create model without pretrained weights
    model = config["model_fn"](weights=None)

    # Replace classifier head for correct num_classes
    if name == "resnet34":
        model.fc = nn.Linear(config["num_features"], num_classes)
    elif name == "densenet121":
        model.classifier = nn.Linear(config["num_features"], num_classes)
    elif name == "efficientnet_b0":
        model.classifier[1] = nn.Linear(config["num_features"], num_classes)
    elif name == "vit_b_16":
        model.heads[0] = nn.Linear(config["num_features"], num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Move to device
    if device is not None:
        model = model.to(device)

    model.eval()

    return model, config["num_features"]


def find_checkpoint_for_backbone(name: str, checkpoint_dir: str) -> Optional[str]:
    """
    Find a checkpoint file for the given backbone name in the checkpoint directory.

    Searches for patterns like: best_{name}.pth, final_{name}.pth, {name}.pth
    """
    checkpoint_path = Path(checkpoint_dir)
    name_lower = name.lower()

    # Common checkpoint naming patterns
    patterns = [
        f"best_{name_lower}.pth",
        f"final_{name_lower}.pth",
        f"{name_lower}.pth",
        f"best_{name_lower}_*.pth",
    ]

    for pattern in patterns:
        matches = list(checkpoint_path.glob(pattern))
        if matches:
            return str(matches[0])

    return None


def create_all_backbones(
    backbone_names: List[str],
    num_classes: int,
    device: torch.device = None,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[List[nn.Module], List[int]]:
    """
    Create all frozen backbones.

    If checkpoint_dir is provided, attempts to load fine-tuned checkpoints.
    Falls back to ImageNet pretrained weights if checkpoint not found.

    Args:
        backbone_names: List of backbone names
        num_classes: Number of output classes
        device: Device to load models onto
        checkpoint_dir: Optional path to directory containing checkpoints

    Returns:
        Tuple of (list of models, list of feature dimensions)
    """
    models_list = []
    feature_dims = []
    total = len(backbone_names)

    for index, name in enumerate(backbone_names, 1):
        checkpoint_path = None
        source = "ImageNet pretrained"

        # Try to find checkpoint if directory provided
        if checkpoint_dir:
            checkpoint_path = find_checkpoint_for_backbone(name, checkpoint_dir)

        if checkpoint_path:
            # Load from fine-tuned checkpoint
            model, feat_dim = load_frozen_backbone_from_checkpoint(
                name, checkpoint_path, num_classes, device
            )
            source = f"checkpoint: {Path(checkpoint_path).name}"
        else:
            # Fall back to ImageNet pretrained
            model, feat_dim = create_frozen_backbone(name, num_classes, device)
            if checkpoint_dir:
                source = "ImageNet pretrained (no checkpoint found)"

        models_list.append(model)
        feature_dims.append(feat_dim)

        # Rich console output
        status = color(f"[{index}/{total}]", Colors.BRIGHT_BLACK)
        name_str = color(name, Colors.BOLD, Colors.WHITE)
        features_str = color(f"{feat_dim:,} features", Colors.CYAN)
        source_str = color(
            source, Colors.BRIGHT_GREEN if checkpoint_path else Colors.YELLOW
        )
        print(f"    {status} [OK] Loaded {name_str} ({features_str})")
        print(f"           Source: {source_str}")

    return models_list, feature_dims


# Default backbones for SEDGE
DEFAULT_BACKBONES = ["resnet34", "densenet121", "efficientnet_b0", "vit_b_16"]
