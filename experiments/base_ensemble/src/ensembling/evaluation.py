"""Core evaluation engine for ensemble models."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from ..data import CLASS_NAMES, get_val_transform
from .inference import ensemble_predict_extended, load_image
from .metrics import calculate_ece, calculate_oracle_accuracy, get_per_class_accuracy


@dataclass
class EvaluationResult:
    """Comprehensive results from an ensemble evaluation."""

    overall_accuracy: float
    oracle_accuracy: float
    ece: float
    avg_disagreement: float
    avg_agreement_ratio: float
    per_class_ensemble: Dict[str, Dict[str, Any]]
    per_class_individuals: Dict[
        str, Dict[str, Dict[str, Any]]
    ]  # model_name -> class_name -> stats
    individual_accuracies: Dict[str, float]
    total_images: int


def evaluate_ensemble_dataset(
    models: List[nn.Module],
    model_names: List[str],
    data_dir: str,
    device: torch.device,
    split: Optional[str] = "val",
    class_names: Optional[List[str]] = None,
    progress_bar: bool = True,
    weights: Optional[List[float]] = None,
) -> EvaluationResult:
    """Evaluate ensemble on a dataset split and return comprehensive metrics."""
    if class_names is None:
        class_names = CLASS_NAMES
    # Handle split=None for direct class folders
    if split is None or split == "":
        eval_dir = Path(data_dir)
    else:
        eval_dir = Path(data_dir) / split
    transform = get_val_transform()

    # Trackers
    all_ensemble_preds = []
    all_individual_preds = [[] for _ in range(len(models))]
    all_targets = []
    all_ensemble_probs = []

    all_disagreements = []
    all_agreement_ratios = []

    # Class to index mapping
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # Collect all image paths
    image_paths = []
    image_targets = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    for class_name in class_names:
        class_dir = eval_dir / class_name
        if not class_dir.exists():
            continue

        for f in class_dir.iterdir():
            if f.suffix.lower() in image_extensions:
                image_paths.append(f)
                image_targets.append(class_to_idx[class_name])

    if not image_paths:
        raise ValueError(f"No images found in {eval_dir}")

    # Main evaluation loop
    iterator = range(len(image_paths))
    if progress_bar:
        iterator = tqdm(iterator, desc=f"Evaluating Ensemble ({split})")

    for i in iterator:
        image_path = str(image_paths[i])
        target_idx = image_targets[i]

        # Load and predict
        image_tensor = load_image(image_path, transform)
        result = ensemble_predict_extended(
            models, model_names, image_tensor, device, class_names, weights=weights
        )

        # Ensemble prediction index
        ensemble_top_idx = torch.argmax(result.ensemble_probs, dim=1).item()
        all_ensemble_preds.append(ensemble_top_idx)
        all_ensemble_probs.append(result.ensemble_probs[0, ensemble_top_idx].item())
        all_targets.append(target_idx)

        # Individual predictions
        for m_idx, (model_class, _) in enumerate(result.individual_predictions):
            pred_idx = class_to_idx.get(model_class, -1)
            all_individual_preds[m_idx].append(pred_idx)

        # Agreement metrics
        all_disagreements.append(result.disagreement)
        all_agreement_ratios.append(result.agreement_ratio)

    # Convert to tensors for metrics
    ensemble_preds_t = torch.tensor(all_ensemble_preds)
    targets_t = torch.tensor(all_targets)
    ensemble_probs_t = torch.tensor(all_ensemble_probs)

    # Calculate metrics
    overall_acc = (ensemble_preds_t == targets_t).float().mean().item() * 100

    # Oracle Accuracy
    individual_preds_t = [torch.tensor(p) for p in all_individual_preds]
    oracle_acc = calculate_oracle_accuracy(individual_preds_t, targets_t) * 100

    # ECE
    accuracies_t = ensemble_preds_t == targets_t
    ece = calculate_ece(ensemble_probs_t, accuracies_t)

    # Per-Class Ensemble
    per_class_ensemble = get_per_class_accuracy(
        ensemble_preds_t, targets_t, class_names
    )

    # Per-Class Individuals
    per_class_individuals = {}
    individual_accuracies = {}
    for m_idx, name in enumerate(model_names):
        m_preds_t = torch.tensor(all_individual_preds[m_idx])
        per_class_individuals[name] = get_per_class_accuracy(
            m_preds_t, targets_t, class_names
        )
        individual_accuracies[name] = (
            m_preds_t == targets_t
        ).float().mean().item() * 100

    return EvaluationResult(
        overall_accuracy=overall_acc,
        oracle_accuracy=oracle_acc,
        ece=ece,
        avg_disagreement=sum(all_disagreements) / len(all_disagreements),
        avg_agreement_ratio=sum(all_agreement_ratios) / len(all_agreement_ratios),
        per_class_ensemble=per_class_ensemble,
        per_class_individuals=per_class_individuals,
        individual_accuracies=individual_accuracies,
        total_images=len(all_targets),
    )
