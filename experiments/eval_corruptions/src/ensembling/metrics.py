"""Metrics and diagnostic tools for ensemble evaluation."""

from typing import Dict, List, Any
import torch


def calculate_ece(
    confidences: torch.Tensor, accuracies: torch.Tensor, num_bins: int = 15
) -> float:
    """Calculate Expected Calibration Error (ECE).

    Args:
        confidences: Max probabilities from the models [N]
        accuracies: Binary correct/incorrect for each sample [N]
        num_bins: Number of bins for calibration
    """
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = torch.zeros(1, device=confidences.device)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def calculate_oracle_accuracy(
    individual_preds: List[torch.Tensor], targets: torch.Tensor
) -> float:
    """Calculate the accuracy if we always picked the correct model if one exists.

    Args:
        individual_preds: List of prediction tensors (indices) [N]
        targets: Target labels [N]
    """
    if not individual_preds:
        return 0.0

    # Stack predictions: [num_models, batch_size]
    stacked_preds = torch.stack(individual_preds)

    # Check if any model got it right for each sample
    # (stacked_preds == targets) gives [num_models, batch_size]
    # .any(dim=0) gives [batch_size] boolean tensor
    any_correct = (stacked_preds == targets.unsqueeze(0)).any(dim=0)

    return any_correct.float().mean().item()


def get_per_class_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, class_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Calculate per-class accuracy for a set of predictions.

    Returns: Dict[class_name -> {"accuracy": float, "correct": int, "total": int}]
    """
    results = {}
    for i, name in enumerate(class_names):
        class_mask = targets == i
        total = class_mask.sum().item()
        if total > 0:
            correct = (predictions[class_mask] == i).sum().item()
            results[name] = {
                "accuracy": 100 * correct / total,
                "correct": correct,
                "total": total,
            }
        else:
            results[name] = {"accuracy": 0.0, "correct": 0, "total": 0}

    return results
