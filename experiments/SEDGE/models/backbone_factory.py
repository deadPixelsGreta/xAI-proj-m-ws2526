"""Factory functions to create frozen pretrained backbones for SEDGE."""

from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision import models

from experiments.SEDGE.training.console_ui import ConsoleUI


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


def create_all_backbones(
    backbone_names: List[str],
    num_classes: int,
    device: torch.device = None,
) -> Tuple[List[nn.Module], List[int]]:
    """
    Create all frozen pretrained backbones.

    Returns:
        Tuple of (list of models, list of feature dimensions)
    """
    models_list = []
    feature_dims = []
    total = len(backbone_names)

    for index, name in enumerate(backbone_names, 1):
        model, feat_dim = create_frozen_backbone(name, num_classes, device)
        models_list.append(model)
        feature_dims.append(feat_dim)

        # Rich console output
        ConsoleUI.backbone_loaded(name, feat_dim, index, total)

    return models_list, feature_dims


# Default backbones for SEDGE
DEFAULT_BACKBONES = ["resnet34", "densenet121", "efficientnet_b0", "vit_b_16"]
