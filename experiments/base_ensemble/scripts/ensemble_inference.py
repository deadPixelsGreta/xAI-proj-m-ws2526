#!/usr/bin/env python3
import argparse
import sys
import torch
from pathlib import Path
from typing import List, Optional

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
from experiments.base_ensemble.src.data import CLASS_NAMES
from experiments.base_ensemble.src.ensembling import (
    run_inference,
    evaluate_ensemble_dataset,
    EvaluationResult,
)
from experiments.base_ensemble.src.ensembling.inference import (
    find_default_checkpoints,
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
    parser.add_argument(
        "--eval-individuals",
        "--eval_individuals",
        action="store_true",
        help="Evaluate individual models along with ensemble on the target set (for fair comparison)",
    )

    # Model arguments
    parser.add_argument(
        "--checkpoints", nargs="+", default=None, help="Paths to model checkpoint files"
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use weighted averaging instead of simple mean. "
        "Weights are taken from --weights or automatically from checkpoint validation accuracy.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Manual weights for each model (must match number of checkpoints)",
    )
    parser.add_argument(
        "--weight-power",
        type=float,
        default=1.0,
        help="Power to raise weights to before normalization (higher values amplify the best model)",
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
        choices=["val", "test", ""],
        help="Dataset split to evaluate on (val or test)",
    )

    return parser.parse_args()


def report_and_log_results(
    result: EvaluationResult,
    model_names: List[str],
    wandb_enabled: bool = False,
    wandb_project: Optional[str] = None,
    split: str = "val",
    data_dir: str = "",
):
    """Report results to console and optional W&B."""
    print("\n" + "=" * 60)
    print(f" Ensemble Evaluation on {split.capitalize()} Set")
    print("=" * 60)

    # Initialize W&B
    if wandb_enabled and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project or "imagenet-subset-ensemble",
            name=f"ensemble-eval-{split}-{len(model_names)}models",
            config={
                "num_models": len(model_names),
                "model_names": model_names,
                "data_dir": data_dir,
                "split": split,
            },
        )
        print(f"   W&B logging enabled: {wandb.run.url}")

    # Per-Class Comparison Table
    print("\n Per-Class Accuracy Comparison:")
    header = f"   {'Class':20s} | {'Ensemble':8s} | " + " | ".join(
        [f"{name[:12]:12s}" for name in model_names]
    )
    print(header)
    print("-" * len(header))

    for class_name in CLASS_NAMES:
        ens_stats = result.per_class_ensemble.get(class_name, {"accuracy": 0})
        row = f"   {class_name:20s} | {ens_stats['accuracy']:7.1f}% | "

        individual_accs = []
        for name in model_names:
            m_stats = result.per_class_individuals[name].get(
                class_name, {"accuracy": 0}
            )
            row += f"{m_stats['accuracy']:11.1f}% | "
            individual_accs.append(m_stats["accuracy"])

        print(row)

    print("-" * len(header))

    # Overall Metrics
    print("\n Performance Metrics:")
    print("-" * 50)
    print(f"   Overall Ensemble Accuracy: {result.overall_accuracy:5.2f}%")
    print(
        f"   Oracle Accuracy:           {result.oracle_accuracy:5.2f}% (theoretical max)"
    )
    print(f"   Expected Calibration Error: {result.ece:5.4f}")
    print(f"   Average Disagreement:      {result.avg_disagreement:5.4f}")
    print(f"   Average Agreement Ratio:   {result.avg_agreement_ratio * 100:5.1f}%")
    print(f"   Total Images:              {result.total_images}")

    print("\n Model Comparison (Overall):")
    print("-" * 50)
    for name in model_names:
        print(f"   {name:25s} {result.individual_accuracies[name]:5.2f}%")
    print(f"   {'ENSEMBLE':25s} {result.overall_accuracy:5.2f}% *")

    # W&B Logging
    if wandb_enabled and WANDB_AVAILABLE:
        # Summary metrics
        wandb.log(
            {
                "ensemble/accuracy": result.overall_accuracy,
                "ensemble/oracle_accuracy": result.oracle_accuracy,
                "ensemble/ece": result.ece,
                "ensemble/avg_disagreement": result.avg_disagreement,
                "ensemble/avg_agreement_ratio": result.avg_agreement_ratio * 100,
                "ensemble/total_images": result.total_images,
            }
        )

        # Detailed per-class table
        columns = ["class", "ensemble_acc"] + [f"{n}_acc" for n in model_names]
        data = []
        for cls in CLASS_NAMES:
            row = [cls, result.per_class_ensemble[cls]["accuracy"]]
            for n in model_names:
                row.append(result.per_class_individuals[n][cls]["accuracy"])
            data.append(row)

        wandb.log(
            {
                "ensemble/detailed_class_comparison": wandb.Table(
                    columns=columns, data=data
                )
            }
        )

        wandb.finish()
        print("\n   W&B logging complete!")


def evaluate_ensemble(
    models,
    model_names,
    data_dir,
    device,
    checkpoint_dir=None,
    split="val",
    wandb_enabled=True,
    wandb_project=None,
    eval_individuals=False,
    weights=None,
):
    """Wrapper to run modular evaluation and report results."""
    result = evaluate_ensemble_dataset(
        models=models,
        model_names=model_names,
        data_dir=data_dir,
        device=device,
        split=None if split == "" else split,
        progress_bar=True,
        weights=weights,
    )

    report_and_log_results(
        result=result,
        model_names=model_names,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        split=split if split else "direct",
        data_dir=str(data_dir),
    )

    return result.overall_accuracy


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
    val_accs = []
    for path in checkpoint_paths:
        model, name, val_acc = load_checkpoint(path, device)
        models.append(model)
        model_names.append(name)
        val_accs.append(val_acc)
        print(f"   ✓ {name:20s} (val acc: {val_acc:.1f}%) from {Path(path).name}")

    # Process weights if requested
    weights = None
    if args.weighted:
        if args.weights:
            if len(args.weights) != len(models):
                print(
                    f"Error: Number of weights ({len(args.weights)}) does not match number of models ({len(models)})"
                )
                sys.exit(1)
            weights = args.weights
        else:
            # Automatic weighting from checkpoint validation accuracy
            # Normalize to 0-1 range if they are percentages (common)
            raw_weights = torch.tensor(val_accs, dtype=torch.float32)
            if raw_weights.max() > 1.0:
                raw_weights = raw_weights / 100.0

            if args.weight_power != 1.0:
                # Apply power transformation to amplify differences in [0, 1] range
                # This makes the "best" model much more dominant without hitting 'inf'
                transformed = torch.pow(raw_weights, args.weight_power)
                weights = transformed.tolist()
                print(
                    f"\n   Applying weight-power {args.weight_power} (normalized to [0,1]):"
                )
                for name, w in zip(model_names, weights):
                    print(f"     {name:25s}: {w:.4f}")
            else:
                weights = raw_weights.tolist()

            print(
                f"\n   Using automatic weights based on val_acc: {[round(w, 4) for w in weights]}"
            )

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
            eval_individuals=args.eval_individuals,
            weights=weights,
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
            weights=weights,
        )

    # Directory inference
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_files = [
            f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"\nNo images found in {args.image_dir}")
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
                weights=weights,
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
