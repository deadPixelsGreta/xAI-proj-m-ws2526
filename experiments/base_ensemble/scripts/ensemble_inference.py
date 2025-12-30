#!/usr/bin/env python3
"""CLI to run ensemble inference/evaluation on ImageNetSubset-style data."""

import argparse
import sys
from pathlib import Path

import torch

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from experiments.base_ensemble.src.utils import get_device
from experiments.base_ensemble.src.utils.device import get_device_name
from experiments.base_ensemble.src.utils.arguments_parsing import (
    add_common_args,
    add_wandb_args,
)
from experiments.base_ensemble.src.models import load_checkpoint
from experiments.base_ensemble.src.data import CLASS_NAMES, get_val_transform
from experiments.base_ensemble.src.ensembling import run_inference
from experiments.base_ensemble.src.ensembling.inference import (
    find_default_checkpoints,
    ensemble_predict_extended,
    load_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensemble Inference for ImageNetSubset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Common arguments (data-dir, checkpoint-dir)
    add_common_args(parser)

    # Input arguments
    parser.add_argument(
        "--image", type=str, default=None, help="Path to a single image for inference"
    )
    parser.add_argument(
        "--image-dir",
        "--image_dir",
        type=str,
        default=None,
        help="Path to directory of images for batch inference",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate ensemble on entire validation set",
    )

    # Model arguments
    parser.add_argument(
        "--checkpoints", nargs="+", default=None, help="Paths to model checkpoint files"
    )

    # Output arguments
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top predictions to show"
    )
    parser.add_argument(
        "--show-individual",
        "--show_individual",
        action="store_true",
        help="Show individual model predictions",
    )
    parser.add_argument(
        "--no-diagnostics",
        "--no_diagnostics",
        action="store_true",
        help="Hide ensemble diagnostic metrics (disagreement, agreement ratio)",
    )

    # Wandb arguments
    add_wandb_args(parser, default_project="imagenet-subset-ensemble")

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate on (val or test)",
    )

    return parser.parse_args()


def evaluate_ensemble(
    models,
    model_names,
    data_dir,
    device,
    checkpoint_dir=None,
    split="val",
    wandb_enabled=False,
    wandb_project=None,
):
    """Evaluate ensemble on entire validation or test set with optional W&B logging."""
    eval_dir = Path(data_dir) / split
    transform = get_val_transform()  # Same transform for val and test
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")

    print("\n" + "=" * 60)
    print(f"Ensemble Evaluation on {split.capitalize()} Set")
    print("=" * 60)

    # Initialize W&B if enabled
    if wandb_enabled and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project or "imagenet-subset-ensemble",
            name=f"ensemble-eval-{split}-{len(models)}models",
            config={
                "num_models": len(models),
                "model_names": model_names,
                "data_dir": str(data_dir),
                "split": split,
            },
        )
        print(f"   W&B logging enabled: {wandb.run.url}")

    total_correct = 0
    total_images = 0
    per_class_correct = {name: 0 for name in CLASS_NAMES}
    per_class_total = {name: 0 for name in CLASS_NAMES}

    # Track disagreement metrics
    all_disagreements = []
    all_agreement_ratios = []

    for class_name in CLASS_NAMES:
        class_dir = eval_dir / class_name
        if not class_dir.exists():
            continue

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_files = [
            f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions
        ]

        class_correct = 0
        for image_file in image_files:
            # Load and predict with extended metrics
            image_tensor = load_image(str(image_file), transform)
            result = ensemble_predict_extended(
                models, model_names, image_tensor, device
            )

            # Track metrics
            all_disagreements.append(result.disagreement)
            all_agreement_ratios.append(result.agreement_ratio)

            if result.predicted_class == class_name:
                class_correct += 1
                total_correct += 1

            total_images += 1

        per_class_correct[class_name] = class_correct
        per_class_total[class_name] = len(image_files)

    # Print per-class results
    print("\n Per-Class Accuracy:")
    print("-" * 50)
    per_class_accuracies = {}
    for class_name in CLASS_NAMES:
        total = per_class_total[class_name]
        correct = per_class_correct[class_name]
        if total > 0:
            acc = 100 * correct / total
            per_class_accuracies[class_name] = acc
            bar = "[" + "#" * int(acc / 5) + "-" * (20 - int(acc / 5)) + "]"
            print(f"   {class_name:20s} {bar} {acc:5.1f}% ({correct}/{total})")

    # Print overall results
    overall_acc = 100 * total_correct / total_images if total_images > 0 else 0
    avg_disagreement = (
        sum(all_disagreements) / len(all_disagreements) if all_disagreements else 0
    )
    avg_agreement = (
        sum(all_agreement_ratios) / len(all_agreement_ratios)
        if all_agreement_ratios
        else 0
    )

    print("\n" + "-" * 50)
    print(
        f"Overall Ensemble Accuracy: {overall_acc:.2f}% ({total_correct}/{total_images})"
    )
    print(f"Average Disagreement:      {avg_disagreement:.4f}")
    print(f"Average Agreement Ratio:   {avg_agreement * 100:.1f}%")

    # Get individual model accuracies for comparison
    individual_accs = {}
    print("\n Model Comparison:")
    print("-" * 50)
    for name in model_names:
        # Get individual model accuracy from checkpoint
        # Handle seed variants by looking for the exact file name if possible,
        # but here we try to find the match within the provided checkpoint_dir
        pattern = f"best_{name}.pth"
        checkpoint_path = checkpoint_dir / pattern

        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            individual_acc = ckpt.get("val_acc", 0)
            individual_accs[name] = individual_acc
            print(f"   {name:25s} {individual_acc:.2f}%")
        else:
            # Fallback for seed-specific filenames if exact match not found
            alt_paths = list(checkpoint_dir.glob(f"best_{name}_seed*.pth"))
            if alt_paths:
                checkpoint_path = alt_paths[0]
                ckpt = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                individual_acc = ckpt.get("val_acc", 0)
                individual_accs[name] = individual_acc
                print(
                    f"   {name:25s} {individual_acc:.2f}% (from {checkpoint_path.name})"
                )
            else:
                print(f"   {name:25s} [Accuracy not found in {checkpoint_dir}]")

    print(f"   {'ENSEMBLE':25s} {overall_acc:.2f}% *")

    # Log to W&B
    if wandb_enabled and WANDB_AVAILABLE:
        # Log summary metrics
        wandb.log(
            {
                "ensemble/accuracy": overall_acc,
                "ensemble/avg_disagreement": avg_disagreement,
                "ensemble/avg_agreement_ratio": avg_agreement * 100,
                "ensemble/total_images": total_images,
                "ensemble/num_models": len(models),
            }
        )

        # Log per-class accuracy table
        class_table = wandb.Table(
            columns=["class", "accuracy", "correct", "total"],
            data=[
                [
                    cls,
                    per_class_accuracies.get(cls, 0),
                    per_class_correct[cls],
                    per_class_total[cls],
                ]
                for cls in CLASS_NAMES
            ],
        )
        wandb.log({"ensemble/per_class_accuracy": class_table})

        # Log model comparison
        model_table = wandb.Table(
            columns=["model", "accuracy", "is_ensemble"],
            data=[[name, acc, False] for name, acc in individual_accs.items()]
            + [["ENSEMBLE", overall_acc, True]],
        )
        wandb.log({"ensemble/model_comparison": model_table})

        # Log disagreement histogram
        wandb.log(
            {
                "ensemble/disagreement_histogram": wandb.Histogram(all_disagreements),
                "ensemble/agreement_histogram": wandb.Histogram(all_agreement_ratios),
            }
        )

        # Summary metrics for quick view in run list
        wandb.run.summary["ensemble/accuracy"] = overall_acc
        wandb.run.summary["ensemble/avg_disagreement"] = avg_disagreement
        wandb.run.summary["ensemble/avg_agreement_ratio"] = avg_agreement * 100

        if individual_accs:
            best_single = max(individual_accs.values())
            improvement = overall_acc - best_single
            wandb.run.summary["ensemble/improvement_over_best_single"] = improvement
            wandb.run.summary["ensemble/best_single_accuracy"] = best_single

        wandb.finish()
        print("\n   W&B logging complete!")

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
        evaluate_ensemble(
            models,
            model_names,
            args.data_dir,
            device,
            checkpoint_dir=args.checkpoint_dir,
            split=args.split,
            wandb_enabled=args.wandb,
            wandb_project=args.wandb_project,
        )

    # Single image inference
    elif args.image:
        run_inference(
            models=models,
            model_names=model_names,
            image_path=args.image,
            device=device,
            top_k=args.top_k,
            show_individual=args.show_individual,
            show_diagnostics=not args.no_diagnostics,
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
            result = run_inference(
                models=models,
                model_names=model_names,
                image_path=str(image_file),
                device=device,
                top_k=args.top_k,
                show_individual=args.show_individual,
                show_diagnostics=not args.no_diagnostics,
            )

            if ground_truth:
                top_prediction = result["predicted_class"]
                if top_prediction == ground_truth:
                    correct += 1
                total += 1

        if ground_truth and total > 0:
            print(
                f"\n Ensemble Accuracy on {ground_truth}: {100 * correct / total:.1f}% ({correct}/{total})"
            )

    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
