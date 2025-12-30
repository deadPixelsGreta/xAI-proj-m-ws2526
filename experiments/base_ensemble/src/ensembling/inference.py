"""Ensemble inference helpers for classification models."""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..data import CLASS_NAMES, get_val_transform


def _get_model_probabilities(
    models: List[nn.Module], image_tensor: torch.Tensor, device: torch.device
) -> List[torch.Tensor]:
    """Internal helper to get softmax probabilities from all models."""
    image_tensor = image_tensor.to(device)
    all_probs = []
    with torch.no_grad():
        for model in models:
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs)
    return all_probs


def ensemble_predict(
    models: List[nn.Module],
    image_tensor: torch.Tensor,
    device: torch.device,
    return_individual: bool = False,
    weights: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """Average softmax probabilities across models for image tensor(s)."""
    all_probs = _get_model_probabilities(models, image_tensor, device)
    stacked_probs = torch.stack(all_probs, dim=0)

    if weights is not None:
        weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
        weight_tensor = F.softmax(weight_tensor, dim=0)
        ensemble_probs = (stacked_probs * weight_tensor.view(-1, 1, 1)).sum(dim=0)
    else:
        ensemble_probs = torch.mean(stacked_probs, dim=0)

    return ensemble_probs, (all_probs if return_individual else None)


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
    class_names: Optional[List[str]] = None,
) -> EnsembleResult:
    """Run ensemble prediction with extended diagnostic metrics."""
    if class_names is None:
        class_names = CLASS_NAMES

    all_probs = _get_model_probabilities(models, image_tensor, device)

    # Stack and average (assuming batch size 1 for these specific metrics currently)
    stacked_probs = torch.stack(all_probs, dim=0)
    ensemble_probs = torch.mean(stacked_probs, dim=0)

    # Get ensemble prediction
    top_prob, top_idx = torch.max(ensemble_probs, dim=1)
    predicted_class = class_names[top_idx.item()]
    confidence = top_prob.item() * 100

    # Compute disagreement
    predicted_class_probs = stacked_probs[:, 0, top_idx.item()]
    disagreement = torch.var(predicted_class_probs).item()

    # Compute individual results
    individual_predictions = []
    agreement_count = 0
    for i, probs in enumerate(all_probs):
        model_top_prob, model_top_idx = torch.max(probs, dim=1)
        model_class = class_names[model_top_idx.item()]
        model_conf = model_top_prob.item() * 100
        individual_predictions.append((model_class, model_conf))
        if model_class == predicted_class:
            agreement_count += 1

    return EnsembleResult(
        predicted_class=predicted_class,
        confidence=confidence,
        ensemble_probs=ensemble_probs,
        individual_probs=all_probs,
        disagreement=disagreement,
        agreement_ratio=agreement_count / len(models),
        individual_predictions=individual_predictions,
    )


def get_top_predictions(
    probs: torch.Tensor, top_k: int = 5, class_names: Optional[List[str]] = None
) -> List[List[Tuple[str, float]]]:
    """Return top-k class names and probabilities for a batch of tensors.

    Returns: List[List[(class_name, confidence_percentage)]]
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Ensure 2D: [Batch, Classes]
    if probs.ndim == 1:
        probs = probs.unsqueeze(0)

    top_k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    batch_results = []
    top_probs_np = top_probs.cpu().numpy()
    top_indices_np = top_indices.cpu().numpy()

    for i in range(probs.size(0)):
        img_results = []
        for prob, idx in zip(top_probs_np[i], top_indices_np[i]):
            img_results.append((class_names[idx], float(prob) * 100))
        batch_results.append(img_results)

    return batch_results


def load_image(image_path: str, transform: Optional[Any] = None) -> torch.Tensor:
    """Load and transform image, returning a 1xCxHxW tensor.

    Raises FileNotFoundError if image_path does not exist.
    """
    if transform is None:
        transform = get_val_transform()

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    try:
        image = Image.open(path).convert("RGB")
        return transform(image).unsqueeze(0)
    except Exception as e:
        raise RuntimeError(f"Error loading image {image_path}: {e}")


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


class EnsembleReporter:
    """Handles console reporting for ensemble results."""

    @staticmethod
    def report_prediction(
        image_path: str, predictions: List[Tuple[str, float]], top_k: int
    ):
        print(f"\n Image: {image_path}")
        print("-" * 60)
        print(f" Ensemble Prediction (top {top_k}):")
        for i, (class_name, confidence) in enumerate(predictions, 1):
            bar = (
                "[" + "#" * int(confidence / 5) + "-" * (20 - int(confidence / 5)) + "]"
            )
            print(f"   {i}. {class_name:20s} {bar} {confidence:5.1f}%")

    @staticmethod
    def report_diagnostics(result: EnsembleResult, models_count: int):
        print("\n Ensemble Diagnostics:")
        print(f"   Confidence Score:     {result.confidence:5.1f}%")
        print(f"   Ensemble Disagreement: {result.disagreement:.4f}")

        if result.disagreement < 0.01:
            label = "Very Low (models agree strongly)"
        elif result.disagreement < 0.05:
            label = "Low (minor differences)"
        elif result.disagreement < 0.10:
            label = "Moderate (some uncertainty)"
        else:
            label = "High (significant disagreement)"

        print(f"   Disagreement Level:   {label}")
        print(
            f"   Model Agreement:       {result.agreement_ratio * 100:.0f}% ({int(result.agreement_ratio * models_count)}/{models_count} models)"
        )

    @staticmethod
    def report_individual(model_names: List[str], result: EnsembleResult):
        print("\n Individual Model Predictions:")
        for model_name, (pred_class, pred_conf) in zip(
            model_names, result.individual_predictions
        ):
            agree_marker = "✓" if pred_class == result.predicted_class else "✗"
            print(
                f"   {agree_marker} {model_name:25s} → {pred_class:20s} ({pred_conf:5.1f}%)"
            )


def run_inference(
    models: List[nn.Module],
    model_names: List[str],
    image_path: str,
    device: torch.device,
    top_k: int = 5,
    show_individual: bool = False,
    show_diagnostics: bool = True,
    transform: Optional[Any] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run ensemble inference for one image path and return results."""
    # Load and preprocess image
    try:
        image_tensor = load_image(image_path, transform)
    except Exception as e:
        print(f"Error: {e}")
        return {}

    # Run core calculation
    result = ensemble_predict_extended(
        models, model_names, image_tensor, device, class_names
    )
    all_predictions = get_top_predictions(result.ensemble_probs, top_k, class_names)
    predictions = all_predictions[0]  # run_inference assumes batch size 1

    # Handle reporting (Decoupled)
    reporter = EnsembleReporter()
    reporter.report_prediction(image_path, predictions, top_k)

    if show_diagnostics:
        reporter.report_diagnostics(result, len(models))

    if show_individual or show_diagnostics:
        reporter.report_individual(model_names, result)

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
