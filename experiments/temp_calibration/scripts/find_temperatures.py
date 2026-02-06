#!/usr/bin/env python3
"""
Finds the optimal Temperature (T) for each expert model using Expected Calibration Error (ECE)
or Negative Log Likelihood (NLL) optimization on a validation subset.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# Add project root to path
def setup_path():
    markers = {".git", "requirements.txt"}
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in markers):
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    return None


setup_path()

from experiments.base_ensemble.src.utils.device import get_device, set_seed
from experiments.base_ensemble.src.models import load_checkpoint
from experiments.base_ensemble.src.data import get_val_transform
from experiments.mixture_of_experts.src.dataset import get_router_datasets


def ece_score(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def find_best_temp(model, val_loader, device, name):
    """
    Finds optimal temperature T for a single model by minimizing NLL on validation set.
    """
    model.eval()
    logits_list = []
    labels_list = []

    # 1. Collect all logits and labels from the validation set
    print(f"   Collecting logits for {name}...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(targets)

    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)

    # 2. Optimize temperature T
    # Use log_temperature to ensure T is always positive (T = exp(log_T))
    log_temperature = nn.Parameter(torch.zeros(1, device=device))
    optimizer = optim.LBFGS([log_temperature], lr=0.01, max_iter=50)

    criterion = nn.CrossEntropyLoss()

    def eval():
        optimizer.zero_grad()
        loss = criterion(logits / torch.exp(log_temperature), labels)
        loss.backward()
        return loss

    # Calculate pre-calibration metrics
    with torch.no_grad():
        before_nll = criterion(logits, labels).item()
        before_ece = ece_score(torch.softmax(logits, dim=1), labels)

    optimizer.step(eval)

    # Calculate post-calibration metrics
    temperature = torch.exp(log_temperature).item()
    after_nll = criterion(logits / temperature, labels).item()
    after_ece = ece_score(torch.softmax(logits / temperature, dim=1), labels)

    print(f"   Done. T = {temperature:.4f}")
    print(f"   NLL: {before_nll:.4f} -> {after_nll:.4f}")
    print(f"   ECE: {before_ece:.4f} -> {after_ece:.4f}")

    return temperature


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate Models")
    parser.add_argument("--data-dir", "--data_dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        type=str,
        default="experiments/base_ensemble/checkpoints",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiments/temp_calibration/checkpoints/temperatures.pth",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--experts",
        nargs="+",
        default=[
            "best_densenet121.pth",
            "best_resnet34.pth",
            "best_efficientnet_b0.pth",
            "best_vit_b_16.pth",
        ],
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print("\n" + "=" * 60)
    print(" Temperature Calibration Loop")
    print("=" * 60)

    # 1. Setup Data (Use the same split logic as router training to be consistent)
    transform = get_val_transform()
    # We use both router_train and router_val for calibration to have more data (500 samples total)
    # The clean move is to use the full val set since experts are already frozen.
    val_path = Path(args.data_dir) / "val"
    import torchvision.datasets as datasets

    val_ds = datasets.ImageFolder(root=str(val_path), transform=transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 2. Iterate through experts
    results = {}
    checkpoint_dir = Path(args.checkpoint_dir)

    for exp_path in args.experts:
        path = (
            checkpoint_dir / exp_path if not os.path.isabs(exp_path) else Path(exp_path)
        )
        model, name, _ = load_checkpoint(str(path), device)

        t = find_best_temp(model, val_loader, device, name)
        results[name] = t

    # 3. Save results
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, save_path)

    print("\n" + "=" * 60)
    print(f" Calibration Complete! Founds temps: {results}")
    print(f" Saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
