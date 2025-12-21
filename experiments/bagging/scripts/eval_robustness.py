#!/usr/bin/env python3
"""Evaluate models on corrupted and custom datasets."""

import argparse
import os
import sys
from pathlib import Path
import torch
from torchvision import datasets, transforms
import PIL.Image as Image
from tqdm import tqdm


# Add project root to sys.path
def setup_path():
    """Find and add project root to sys.path."""
    markers = {".git", "requirements.txt", "setup.py"}
    current = Path(__file__).resolve().parent

    # Walk up to find the root that contains 'experiments' and ideally '.git'
    project_root = None
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            project_root = parent
            break
        if (parent / "experiments").exists() and any(
            (parent / m).exists() for m in markers
        ):
            project_root = parent
            break

    if project_root:
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        return project_root

    # Fallback
    project_root = current.parents[2]  # scripts -> bagging -> experiments -> root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


_project_root = setup_path()

try:
    from experiments.bagging.src.models import load_checkpoint
    from experiments.bagging.src.utils import get_device
    from experiments.bagging.src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    print(
        "Error: Could not import project modules. Ensure you are running from the project root or scripts directory."
    )
    sys.exit(1)


class Pixelate:
    """Dynamic resolution reduction to simulate low-quality capture."""

    def __init__(self, ratio=0.2):
        self.ratio = ratio

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        size = img.size
        # Downsample
        small = img.resize(
            (int(size[0] * self.ratio), int(size[1] * self.ratio)),
            resample=Image.BILINEAR,
        )
        # Upsample back to original size
        return small.resize(size, resample=Image.NEAREST)


class GaussianNoise:
    """Additive Gaussian noise."""

    def __init__(self, std=0.08):
        self.std = std

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        noise = torch.randn_like(img) * self.std
        return torch.clamp(img + noise, 0, 1)


def get_eval_transform(corruption=None):
    """Factory for evaluation transforms with optional corruptions."""
    base = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]

    if corruption == "pixelate":
        base.append(Pixelate(0.15))  # Stronger pixelation
    elif corruption == "gaussian":
        base.append(GaussianNoise(0.1))  # Significant noise

    # Finalize with conversion and normalization
    if corruption == "gaussian":
        # GaussianNoise already returns a tensor
        base.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    else:
        base.extend(
            [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        )
    return transforms.Compose(base)


def evaluate(model, loader, device):
    """Main evaluation loop."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Testing", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model robustness to corruptions and custom datasets."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory (ImageFolder layout)",
    )
    parser.add_argument(
        "--batch-size", "--batch_size", type=int, default=32, dest="batch_size"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--scenarios",
        type=str,
        default="clean,pixelate,gaussian",
        help="Comma-separated list of scenarios: clean, pixelate, gaussian",
    )
    parser.add_argument(
        "--custom-name",
        type=str,
        default="Custom",
        help="Name for the custom dataset in report",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model (assuming 10 classes as per project specs)
    print(f"Loading model: {args.model}")
    try:
        model, _, _ = load_checkpoint(args.model, device, num_classes=10)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    scenarios = [s.strip() for s in args.scenarios.split(",")]
    results = {}

    print("\nComparing robustness across scenarios...")
    for scenario in scenarios:
        print(f" -> Scenario: {scenario}")

        # Map scenario to corruption
        corruption = None if scenario == "clean" else scenario
        if scenario not in ["clean", "pixelate", "gaussian"]:
            # If it's a custom name not in our predefined list, we treat it as 'clean'
            # (i.e., we evaluate the raw images from the custom directory)
            corruption = None

        transform = get_eval_transform(corruption)

        if not os.path.exists(args.data_dir):
            print(f"    [!] Error: Directory {args.data_dir} not found.")
            continue

        try:
            dataset = datasets.ImageFolder(args.data_dir, transform=transform)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )

            acc = evaluate(model, loader, device)
            results[scenario] = acc
            print(f"    Accuracy: {acc:.2f}%")
        except Exception as e:
            print(f"    [!] Error during evaluation: {e}")

    # Final Report
    print("\n" + "=" * 40)
    print(" ROBUSTNESS EVALUATION REPORT")
    print(f" Model: {Path(args.model).name}")
    print("=" * 40)
    print(f"{'Scenario':<15} | {'Accuracy':<10}")
    print("-" * 30)
    for s, a in results.items():
        display_name = s.capitalize() if s != "clean" else "Clean (Base)"
        print(f"{display_name:<15} | {a:>8.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
