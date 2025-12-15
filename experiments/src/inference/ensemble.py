"""Ensemble inference helpers for classification models."""

from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..data import CLASS_NAMES, get_val_transform
from ..models import SUPPORTED_MODELS, load_checkpoint


def ensemble_predict(
    models: List[nn.Module],
    image_tensor: torch.Tensor,
    device: torch.device,
    return_individual: bool = False,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """Average softmax probabilities across models for one image tensor.

    Returns (ensemble_probs, individual_probs_or_none).
    """
    image_tensor = image_tensor.to(device)
    all_probs = []

    with torch.no_grad():
        for model in models:
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs)

    # Average probabilities
    stacked_probs = torch.stack(all_probs, dim=0)
    ensemble_probs = torch.mean(stacked_probs, dim=0)

    if return_individual:
        return ensemble_probs, all_probs
    return ensemble_probs, None


def get_top_predictions(
    probs: torch.Tensor, top_k: int = 5, class_names: List[str] = None
) -> List[Tuple[str, float]]:
    """Return the top-k class names and probabilities (percentage)."""
    if class_names is None:
        class_names = CLASS_NAMES

    top_probs, top_indices = torch.topk(probs[0], k=min(top_k, len(class_names)))

    predictions = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        predictions.append((class_names[idx], float(prob) * 100))

    return predictions


def load_image(image_path: str, transform=None) -> torch.Tensor:
    """Load an image path, apply the transform (defaults to val), and return a 1xCxHxW tensor."""
    if transform is None:
        transform = get_val_transform()

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def find_default_checkpoints(checkpoint_dir: str = "checkpoints") -> List[str]:
    """Return best_* checkpoint paths under checkpoint_dir for supported models."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []

    checkpoints = []
    for model_name in SUPPORTED_MODELS:
        best_path = checkpoint_path / f"best_{model_name}.pth"
        if best_path.exists():
            checkpoints.append(str(best_path))

    return checkpoints


def run_inference(
    models: List[nn.Module],
    model_names: List[str],
    image_path: str,
    device: torch.device,
    top_k: int = 5,
    show_individual: bool = False,
    transform=None,
) -> List[Tuple[str, float]]:
    """Run ensemble inference for one image path and print top-k results.

    Returns a list of (class_name, confidence_percentage).
    """
    print(f"\nImage: {image_path}")
    print("-" * 50)

    # Load and preprocess image
    if transform is None:
        transform = get_val_transform()
    image_tensor = load_image(image_path, transform)

    # Run ensemble prediction
    ensemble_probs, individual_probs = ensemble_predict(
        models, image_tensor, device, return_individual=show_individual
    )

    # Get ensemble predictions
    predictions = get_top_predictions(ensemble_probs, top_k)

    print(f"Ensemble Prediction (top {top_k}):")
    for i, (class_name, confidence) in enumerate(predictions, 1):
        bar = "█" * int(confidence / 5) + "░" * (20 - int(confidence / 5))
        print(f"   {i}. {class_name:20s} {bar} {confidence:5.1f}%")

    # Show individual model predictions if requested
    if show_individual and individual_probs:
        print(f"\nIndividual Model Predictions:")
        for model_name, probs in zip(model_names, individual_probs):
            top_prob, top_idx = torch.max(probs, dim=1)
            class_name = CLASS_NAMES[top_idx.item()]
            confidence = top_prob.item() * 100
            print(f"   {model_name:20s} -> {class_name:20s} ({confidence:.1f}%)")

    return predictions
