#!/usr/bin/env python3
"""CLI to run ensemble inference/evaluation on ImageNetSubset-style data."""

import argparse
import sys
from pathlib import Path

from experiments.src.utils import get_device
from experiments.src.utils.device import get_device_name
from experiments.src.models import load_checkpoint
from experiments.src.data import CLASS_NAMES, get_val_transform
from experiments.src.inference import run_inference
from experiments.src.inference.ensemble import (
    find_default_checkpoints,
    ensemble_predict,
    load_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensemble Inference for ImageNetSubset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input arguments
    parser.add_argument(
        "--image", type=str, default=None, help="Path to a single image for inference"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to directory of images for batch inference",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate ensemble on entire validation set",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ImageNetSubset",
        help="Dataset directory (for --evaluate mode)",
    )

    # Model arguments
    parser.add_argument(
        "--checkpoints", nargs="+", default=None, help="Paths to model checkpoint files"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "checkpoints"),
        help="Directory to search for default checkpoints",
    )

    # Output arguments
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top predictions to show"
    )
    parser.add_argument(
        "--show-individual",
        action="store_true",
        help="Show individual model predictions",
    )

    return parser.parse_args()


def evaluate_ensemble(models, model_names, data_dir, device):
    """Evaluate ensemble on entire validation set."""
    val_dir = Path(data_dir) / "val"
    transform = get_val_transform()

    print("\n" + "=" * 60)
    print("Ensemble Evaluation on Validation Set")
    print("=" * 60)

    total_correct = 0
    total_images = 0
    per_class_correct = {name: 0 for name in CLASS_NAMES}
    per_class_total = {name: 0 for name in CLASS_NAMES}

    for class_name in CLASS_NAMES:
        class_dir = val_dir / class_name
        if not class_dir.exists():
            continue

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_files = [
            f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions
        ]

        class_correct = 0
        for image_file in image_files:
            # Load and predict
            image_tensor = load_image(str(image_file), transform)
            ensemble_probs, _ = ensemble_predict(models, image_tensor, device)

            # Get prediction
            _, predicted_idx = ensemble_probs[0].max(0)
            predicted_class = CLASS_NAMES[predicted_idx.item()]

            if predicted_class == class_name:
                class_correct += 1
                total_correct += 1

            total_images += 1

        per_class_correct[class_name] = class_correct
        per_class_total[class_name] = len(image_files)

    # Print per-class results
    print("\n Per-Class Accuracy:")
    print("-" * 50)
    for class_name in CLASS_NAMES:
        total = per_class_total[class_name]
        correct = per_class_correct[class_name]
        if total > 0:
            acc = 100 * correct / total
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"   {class_name:20s} {bar} {acc:5.1f}% ({correct}/{total})")

    # Print overall results
    overall_acc = 100 * total_correct / total_images if total_images > 0 else 0
    print("\n" + "-" * 50)
    print(f"Overall Ensemble Accuracy: {overall_acc:.2f}% ({total_correct}/{total_images})")

    # Compare with individual models
    print(f"\n Model Comparison:")
    print("-" * 50)
    for name in model_names:
        # Get individual model accuracy from checkpoint
        checkpoint_path = Path("checkpoints") / f"best_{name}.pth"
        if checkpoint_path.exists():
            import torch

            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            individual_acc = ckpt.get("val_acc", 0)
            print(f"   {name:20s} {individual_acc:.2f}%")
    print(f"   {'ENSEMBLE':20s} {overall_acc:.2f}% ⭐")

    return overall_acc


def main():
    args = parse_args()

    # Validate inputs
    if args.image is None and args.image_dir is None and not args.evaluate:
        print("Error: Please specify --image, --image-dir, or --evaluate")
        sys.exit(1)

    print("=" * 60)
    print("Ensemble Inference for ImageNetSubset")
    print("=" * 60)

    # Setup device
    device = get_device()
    print(f"\nDevice: {get_device_name(device)}")

    # Find checkpoints
    if args.checkpoints:
        checkpoint_paths = args.checkpoints
    else:
        checkpoint_paths = find_default_checkpoints(args.checkpoint_dir)

    if not checkpoint_paths:
        print("\n No checkpoints found!")
        print("   Please train models first or specify --checkpoints")
        print(f"   Expected checkpoints in: {args.checkpoint_dir}/best_*.pth")
        sys.exit(1)

    print(f"\n Loading {len(checkpoint_paths)} model(s):")

    # Load models
    models = []
    model_names = []
    for path in checkpoint_paths:
        model, name, val_acc = load_checkpoint(path, device)
        models.append(model)
        model_names.append(name)
        print(f"   ✓ {name:20s} (val acc: {val_acc:.1f}%) from {Path(path).name}")

    # Run evaluation mode
    if args.evaluate:
        evaluate_ensemble(models, model_names, args.data_dir, device)

    # Single image inference
    elif args.image:
        run_inference(
            models=models,
            model_names=model_names,
            image_path=args.image,
            device=device,
            top_k=args.top_k,
            show_individual=args.show_individual,
        )

    # Directory inference
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_files = [
            f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"\n❌ No images found in {args.image_dir}")
            sys.exit(1)

        print(f"\n Processing {len(image_files)} images from {args.image_dir}")

        correct = 0
        total = 0

        # Try to get ground truth from parent directory name
        ground_truth = image_dir.name if image_dir.name in CLASS_NAMES else None

        for image_file in sorted(image_files)[:20]:  # Limit to 20 for display
            predictions = run_inference(
                models=models,
                model_names=model_names,
                image_path=str(image_file),
                device=device,
                top_k=args.top_k,
                show_individual=args.show_individual,
            )

            if ground_truth:
                top_prediction = predictions[0][0]
                if top_prediction == ground_truth:
                    correct += 1
                total += 1

        if ground_truth and total > 0:
            print(f"\n Ensemble Accuracy on {ground_truth}: {100 * correct / total:.1f}% ({correct}/{total})")

    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
