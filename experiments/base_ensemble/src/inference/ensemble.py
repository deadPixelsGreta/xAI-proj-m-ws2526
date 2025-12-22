"""Ensemble inference helpers for classification models."""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..data import CLASS_NAMES, get_val_transform


def ensemble_predict(
    models: List[nn.Module],
    image_tensor: torch.Tensor,
    device: torch.device,
    return_individual: bool = False,
    weights: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """Average softmax probabilities across models for one image tensor.

    Args:
        models: List of models to ensemble.
        image_tensor: Input image tensor.
        device: Device to run inference on.
        return_individual: If True, also return individual model probabilities.
        weights: Optional list of weights for weighted averaging (e.g., validation accuracies).
                 If None, uses simple averaging.

    Returns (ensemble_probs, individual_probs_or_none).
    """
    image_tensor = image_tensor.to(device)
    all_probs = []

    with torch.no_grad():
        for model in models:
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs)

    # Stack probabilities: [num_models, batch, num_classes]
    stacked_probs = torch.stack(all_probs, dim=0)

    if weights is not None:
        # Weighted averaging based on provided weights (e.g., val accuracies)
        weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
        weight_tensor = F.softmax(weight_tensor, dim=0)  # Normalize to sum to 1
        ensemble_probs = (stacked_probs * weight_tensor.view(-1, 1, 1)).sum(dim=0)
    else:
        # Simple averaging
        ensemble_probs = torch.mean(stacked_probs, dim=0)

    if return_individual:
        return ensemble_probs, all_probs
    return ensemble_probs, None


@dataclass
class EnsembleResult:
    """Extended ensemble prediction result with diagnostic metrics."""

    predicted_class: str
    confidence: float  # Max probability (0-100%)
    ensemble_probs: torch.Tensor  # Full probability distribution
    individual_probs: List[torch.Tensor]  # Each model's probabilities
    disagreement: float  # Variance in predicted class probability across models
    agreement_ratio: float  # Fraction of models agreeing on top prediction
    individual_predictions: List[Tuple[str, float]]  # Each model's (class, confidence)


def ensemble_predict_extended(
    models: List[nn.Module],
    model_names: List[str],
    image_tensor: torch.Tensor,
    device: torch.device,
) -> EnsembleResult:
    """Run ensemble prediction with extended diagnostic metrics.

    Returns an EnsembleResult with:
    - predicted_class: Top predicted class name
    - confidence: Ensemble confidence (max averaged probability)
    - disagreement: Variance in top-class probability across models (0-1 scale)
    - agreement_ratio: Fraction of models predicting the same class
    - individual_predictions: List of (class_name, confidence) per model
    """
    image_tensor = image_tensor.to(device)
    all_probs = []

    with torch.no_grad():
        for model in models:
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs)

    # Stack and average
    stacked_probs = torch.stack(all_probs, dim=0)  # [num_models, batch, num_classes]
    ensemble_probs = torch.mean(stacked_probs, dim=0)  # [batch, num_classes]

    # Get ensemble prediction
    top_prob, top_idx = torch.max(ensemble_probs, dim=1)
    predicted_class = CLASS_NAMES[top_idx.item()]
    confidence = top_prob.item() * 100

    # Compute disagreement: variance of probability for the predicted class across models
    predicted_class_probs = stacked_probs[:, 0, top_idx.item()]  # [num_models]
    disagreement = torch.var(predicted_class_probs).item()

    # Compute agreement ratio: how many models agree on the top prediction
    individual_predictions = []
    agreement_count = 0
    for i, probs in enumerate(all_probs):
        model_top_prob, model_top_idx = torch.max(probs, dim=1)
        model_class = CLASS_NAMES[model_top_idx.item()]
        model_conf = model_top_prob.item() * 100
        individual_predictions.append((model_class, model_conf))
        if model_class == predicted_class:
            agreement_count += 1

    agreement_ratio = agreement_count / len(models)

    return EnsembleResult(
        predicted_class=predicted_class,
        confidence=confidence,
        ensemble_probs=ensemble_probs,
        individual_probs=all_probs,
        disagreement=disagreement,
        agreement_ratio=agreement_ratio,
        individual_predictions=individual_predictions,
    )


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
    """Return all best_*.pth checkpoint paths under checkpoint_dir.

    Discovers all checkpoints matching the pattern best_*.pth, including
    seed variants like best_resnet18_seed0.pth, best_resnet34_seed1.pth, etc.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []

    # Find all best_*.pth files (includes seed variants)
    checkpoints = sorted(checkpoint_path.glob("best_*.pth"))

    return [str(ckpt) for ckpt in checkpoints]


def run_inference(
    models: List[nn.Module],
    model_names: List[str],
    image_path: str,
    device: torch.device,
    top_k: int = 5,
    show_individual: bool = False,
    show_diagnostics: bool = True,
    transform=None,
) -> Dict[str, Any]:
    """Run ensemble inference for one image path with extended diagnostics.

    Returns a dict with:
    - predictions: List of (class_name, confidence_percentage)
    - confidence: Max ensemble probability
    - disagreement: Variance across model predictions
    - agreement_ratio: Fraction of models agreeing
    - individual: List of (model_name, predicted_class, confidence)
    """
    print(f"\n Image: {image_path}")
    print("-" * 60)

    # Load and preprocess image
    if transform is None:
        transform = get_val_transform()
    image_tensor = load_image(image_path, transform)

    # Run extended ensemble prediction
    result = ensemble_predict_extended(models, model_names, image_tensor, device)

    # Get top-k predictions
    predictions = get_top_predictions(result.ensemble_probs, top_k)

    # Print ensemble predictions
    print(f" Ensemble Prediction (top {top_k}):")
    for i, (class_name, confidence) in enumerate(predictions, 1):
        bar = "[" + "#" * int(confidence / 5) + "-" * (20 - int(confidence / 5)) + "]"
        print(f"   {i}. {class_name:20s} {bar} {confidence:5.1f}%")

    # Print diagnostic metrics
    if show_diagnostics:
        print("\n Ensemble Diagnostics:")
        print(f"   Confidence Score:     {result.confidence:5.1f}%")
        print(f"   Ensemble Disagreement: {result.disagreement:.4f}")

        # Interpret disagreement level
        if result.disagreement < 0.01:
            disagreement_label = "Very Low (models agree strongly)"
        elif result.disagreement < 0.05:
            disagreement_label = "Low (minor differences)"
        elif result.disagreement < 0.10:
            disagreement_label = "Moderate (some uncertainty)"
        else:
            disagreement_label = "High (significant disagreement)"
        print(f"   Disagreement Level:   {disagreement_label}")

        print(
            f"   Model Agreement:       {result.agreement_ratio * 100:.0f}% ({int(result.agreement_ratio * len(models))}/{len(models)} models)"
        )

    # Always show individual model predictions for diagnostics
    if show_individual or show_diagnostics:
        print("\n Individual Model Predictions:")
        for model_name, (pred_class, pred_conf) in zip(
            model_names, result.individual_predictions
        ):
            # Highlight if model disagrees with ensemble
            agree_marker = "✓" if pred_class == result.predicted_class else "✗"
            print(
                f"   {agree_marker} {model_name:25s} → {pred_class:20s} ({pred_conf:5.1f}%)"
            )

    return {
        "predictions": predictions,
        "predicted_class": result.predicted_class,
        "confidence": result.confidence,
        "disagreement": result.disagreement,
        "agreement_ratio": result.agreement_ratio,
        "individual": [
            (name, pred[0], pred[1])
            for name, pred in zip(model_names, result.individual_predictions)
        ],
    }
