#!/usr/bin/env python3
"""CLI to train supported models on ImageNetSubset-style data."""

import argparse
import sys
from datetime import datetime
from pathlib import Path


# Add project root to Python path to ensure 'experiments' module can be found
def setup_path():
    """Find and add project root to sys.path."""
    markers = {".git", "requirements.txt", "setup.py", "pyproject.toml"}
    current = Path(__file__).resolve().parent  # experiments/scripts/

    # Walk up from current directory to find project root
    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in markers):
            project_root = parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            return project_root

    # Fallback: use parent's parent's parent (scripts -> eval_corruptions -> experiments -> project root)
    project_root = (current.parent.parent.parent).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


# Call setup_path and verify
_project_root = setup_path()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from experiments.eval_corruptions.src.utils.device import get_device, set_seed  # noqa: E402
from experiments.eval_corruptions.src.utils.device import get_device_name  # noqa: E402
from experiments.eval_corruptions.src.utils.arguments_parsing import (
    add_common_args,
    add_wandb_args,
    add_model_args,
)
from experiments.eval_corruptions.src.models import SUPPORTED_MODELS, create_model  # noqa: E402
from experiments.eval_corruptions.src.data import create_data_loaders, get_dataset_info  # noqa: E402
from experiments.eval_corruptions.src.training import train, TrainingConfig  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN on ImageNetSubset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (YAML)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (values can be overridden by CLI flags)",
    )

    # Early stopping and checkpoint arguments
    parser.add_argument(
        "--early-stopping-thresh",
        "--early_stopping_thresh",
        type=int,
        default=8,
        help="Number of epochs without improvement before early stopping",
        dest="early_stopping_thresh",
    )
    parser.add_argument(
        "--top-n-checkpoints",
        "--top_n_checkpoints",
        type=int,
        default=2,
        help="Number of best checkpoints to keep",
        dest="top_n_checkpoints",
    )

    parser.add_argument(
        "--freeze-backbone",
        "--freeze_backbone",
        type=bool, 
        default=False, 
        help="Whether to freeze the backbone during training"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="densenet121",
        choices=SUPPORTED_MODELS,
        help="Model architecture",
    )
    parser.add_argument(
        "--no-pretrained",
        "--no_pretrained",
        action="store_true",
        help="Train from scratch without pretrained weights",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        type=str,
        default="ImageNetSubset",
        help="Path to dataset directory",
    )
========
    # Common arguments (data-dir, checkpoint-dir/save-dir)
    add_common_args(parser)

    # Model arguments
    add_model_args(parser, SUPPORTED_MODELS)
>>>>>>>> main:experiments/base_ensemble/scripts/train_single_model.py
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
        dest="batch_size",
    )
    parser.add_argument(
        "--num-workers", 
        "--num_workers",
        type=int, default=4, help="Number of data loader workers"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument(
        "--weight-decay", 
        "--weight_decay", 
        type=float, 
        default=1e-4, 
        help="Weight decay (L2 regularization)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
<<<<<<<< HEAD:experiments/resnets_aug/scripts/train.py

    # Optimizer arguments
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adamw"],
        help="Optimizer type (sgd or adamw)",
    )
    parser.add_argument(
        "--label-smoothing",
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor",
        dest="label_smoothing",
    )

    parser.add_argument(
        "--num-ops",
        "--num_ops",
        type=int,
        default=2,
        help="Number of augmentation operations for RandAugment",
    )
    parser.add_argument(
        "--magnitude",
        type=int,
        default=9,
        help="Magnitude of augmentation operations for RandAugment",
    )
    parser.add_argument(
        "--use-cutout",
        "--use_cutout",
        type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=False,
        help="Enable cutout/random erasing augmentation",
        dest="use_cutout",
    )
    parser.add_argument(
        "--cutout-size",
        "--cutout_size",
        type=int,
        default=32,
        help="Size of cutout region (for random erasing)",
    )
    parser.add_argument(
        "--use-mixup",
        "--use_mixup",
        type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=False,
        help="Enable mixup augmentation",
        dest="use_mixup",
    )
    parser.add_argument(
        "--mixup-alpha",
        "--mixup_alpha",
        type=float,
        default=0.2,
        help="Alpha parameter for mixup augmentation",
        dest="mixup_alpha",
    )
    parser.add_argument(
        "--use-cutmix",
        "--use_cutmix",
        type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=False,
        help="Enable CutMix augmentation",
        dest="use_cutmix",
    )
    parser.add_argument(
        "--cutmix-alpha",
        "--cutmix_alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for CutMix augmentation",
        dest="cutmix_alpha",
    )

    # Output arguments
    parser.add_argument(
        "--save-dir",
        "--save_dir",
========
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (0 to disable)",
        dest="early_stopping_patience",
    )
    parser.add_argument(
        "--no-sota",
        action="store_true",
        help="Disable SOTA data augmentation (TrivialAugmentWide, Mixup, Cutmix)",
    )
    parser.add_argument(
        "--aug-policy",
>>>>>>>> main:experiments/base_ensemble/scripts/train_single_model.py
        type=str,
        default=None,
        choices=["blur", "color", "geometric", "vit"],
        help="Augmentation policy for diversity: blur, color, geometric, or vit (overrides --no-sota if set)",
    )

    # Wandb arguments
<<<<<<<< HEAD:experiments/resnets_aug/scripts/train.py
    parser.add_argument(
        "--wandb", 
        type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=True,
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        "--wandb_project",
        type=str, default="imagenet-subset", help="W&B project name"
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

========
    add_wandb_args(parser)
>>>>>>>> main:experiments/base_ensemble/scripts/train_single_model.py
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply YAML config if provided (CLI flags take precedence)
    if args.config:
        if yaml is None:
            print("PyYAML is not installed. Please install with: pip install pyyaml")
            sys.exit(1)
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {args.config}")
            sys.exit(1)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        def cli_overrides(*opts: str) -> bool:
            """Check if any of the given CLI options were provided."""
            for opt in opts:
                if any(arg == opt or arg.startswith(f"{opt}=") for arg in sys.argv):
                    return True
            return False

        # training section
        tr = cfg.get("training", {})
        if "epochs" in tr and not cli_overrides("--epochs"):
            args.epochs = int(tr["epochs"])
        if "batch_size" in tr and not cli_overrides("--batch-size", "--batch_size"):
            args.batch_size = int(tr["batch_size"])
        if "learning_rate" in tr and not cli_overrides("--lr"):
            args.lr = float(tr["learning_rate"])
        if "momentum" in tr and not cli_overrides("--momentum"):
            args.momentum = float(tr["momentum"])
        if "weight_decay" in tr and not cli_overrides(
            "--weight-decay", "--weight_decay"
        ):
            args.weight_decay = float(tr["weight_decay"])
        if "num_workers" in tr and not cli_overrides("--num-workers", "--num_workers"):
            args.num_workers = int(tr["num_workers"])
<<<<<<<< HEAD:experiments/resnets_aug/scripts/train.py
        # early_stopping_thresh in yaml
        if "early_stopping_thresh" in tr and not cli_overrides("--stop-thresh"):
            args.early_stopping_thresh = int(tr["early_stopping_thresh"])
        # top_n_checkpoints in yaml
        if "top_n_checkpoints" in tr and not cli_overrides("--top-n"):
            args.top_n_checkpoints = int(tr["top_n_checkpoints"])
        # freeze_backbone in yaml
        if "freeze_backbone" in tr and not cli_overrides("--freeze-backbone"):
            args.freeze_backbone = bool(tr["freeze_backbone"])
        if "optimizer" in tr and not cli_overrides("--optimizer"):
            args.optimizer = str(tr["optimizer"])
        if "label_smoothing" in tr and not cli_overrides("--label-smoothing"):
========
        if "early_stopping_patience" in tr and not cli_overrides("--patience"):
            args.early_stopping_patience = int(tr["early_stopping_patience"])
        if "amp" in tr and not cli_overrides("--amp"):
            args.amp = bool(tr["amp"])
        if "grad_clip" in tr and not cli_overrides("--grad-clip", "--grad_clip"):
            args.grad_clip = float(tr["grad_clip"])
        if "accum_steps" in tr and not cli_overrides("--accum-steps", "--accum_steps"):
            args.accum_steps = int(tr["accum_steps"])
        if "scheduler" in tr and not cli_overrides("--scheduler"):
            args.scheduler = str(tr["scheduler"])
        if "label_smoothing" in tr and not cli_overrides("--label-smoothing", "--label_smoothing"):
>>>>>>>> main:experiments/base_ensemble/scripts/train_single_model.py
            args.label_smoothing = float(tr["label_smoothing"])

        # augmentation parameters in yaml
        aug = cfg.get("augmentation", {})
        if "num_ops" in aug and not cli_overrides("--num-ops"):
            args.num_ops = int(aug["num_ops"])
        if "magnitude" in aug and not cli_overrides("--magnitude"):
            args.magnitude = int(aug["magnitude"])
        if "use_cutout" in aug and not cli_overrides("--use-cutout"):
            args.use_cutout = bool(aug["use_cutout"])
        if "cutout_size" in aug and not cli_overrides("--cutout-size"):
            args.cutout_size = int(aug["cutout_size"])
        if "use_mixup" in aug and not cli_overrides("--use-mixup"):
            args.use_mixup = bool(aug["use_mixup"])
        if "mixup_alpha" in aug and not cli_overrides("--mixup-alpha"):
            args.mixup_alpha = float(aug["mixup_alpha"])
        
        # model section
        md = cfg.get("model", {})
        if "architecture" in md and not cli_overrides("--model"):
            args.model = str(md["architecture"])
        if "pretrained" in md and not cli_overrides(
            "--no-pretrained", "--no_pretrained"
        ):
            # CLI flag disables pretrained; YAML true => keep pretrained
            pretrained_yaml = bool(md["pretrained"])
            args.no_pretrained = not pretrained_yaml

        # data section
        dd = cfg.get("data", {})
        if "data_dir" in dd and not cli_overrides("--data-dir", "--data_dir"):
            args.data_dir = str(dd["data_dir"])

        # paths section
        pd = cfg.get("paths", {})
        if "checkpoint_dir" in pd and not (
            cli_overrides("--checkpoint-dir", "--checkpoint_dir")
            or cli_overrides("--save-dir", "--save_dir")
        ):
            args.checkpoint_dir = str(pd["checkpoint_dir"])

        # wandb section
        wb = cfg.get("wandb", {})
        if "enabled" in wb and not (
            cli_overrides("--wandb", "--wandb")
            or cli_overrides("--no-wandb", "--no_wandb")
        ):
            args.wandb = bool(wb["enabled"])

        # CLI flag --no-wandb always disables, --wandb always enables
        if cli_overrides("--no-wandb", "--no_wandb"):
            args.wandb = False
        elif cli_overrides("--wandb"):
            args.wandb = True
        if "project" in wb and not cli_overrides("--wandb-project", "--wandb_project"):
            args.wandb_project = str(wb["project"])
        # Optional W&B fields from YAML (no CLI flags defined for these)
        if "run_name" in wb:
            args.wandb_run_name = wb.get("run_name")
        if "entity" in wb and not cli_overrides("--wandb-entity", "--wandb_entity"):
            args.wandb_entity = wb.get("entity")
        if "tags" in wb and not cli_overrides("--wandb-tags", "--wandb_tags"):
            args.wandb_tags = wb.get("tags")
        if "notes" in wb and not cli_overrides("--wandb-notes", "--wandb_notes"):
            args.wandb_notes = wb.get("notes")
        if "mode" in wb and not cli_overrides("--wandb-mode", "--wandb_mode"):
            args.wandb_mode = wb.get("mode")
        if "dir" in wb and not cli_overrides("--wandb-dir", "--wandb_dir"):
            args.wandb_dir = wb.get("dir")
        if "group" in wb and not cli_overrides("--wandb-group", "--wandb_group"):
            args.wandb_group = wb.get("group")
        if "job_type" in wb and not cli_overrides(
            "--wandb-job-type", "--wandb_job_type"
        ):
            args.wandb_job_type = wb.get("job_type")

    # Auto-generate wandb run_name from model and timestamp if not provided
    if args.wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.wandb_run_name = f"{args.model}-{timestamp}"

    print("=" * 60)
    print(f"{args.model.upper()} Training on ImageNetSubset")
    print("=" * 60)

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        print(f"\nRandom seed: {args.seed}")

    # Setup device
    device = get_device()
    print(f"Device: {get_device_name(device)}")

    # Load dataset info
    dataset_info = get_dataset_info(args.data_dir)
    print("\n Dataset Summary:")
    print(f"   Training samples: {dataset_info['train_samples']}")
    print(f"   Validation samples: {dataset_info['val_samples']}")
    print(f"   Classes: {dataset_info['classes']}")
    print(f"   Number of classes: {dataset_info['num_classes']}")

    # Create data loaders
<<<<<<<< HEAD:experiments/resnets_aug/scripts/train.py
    aug_params = {
        'num_ops': args.num_ops,
        'magnitude': args.magnitude,
        'use_cutout': args.use_cutout,
        'cutout_size': args.cutout_size,
        'use_mixup': args.use_mixup,
        'mixup_alpha': args.mixup_alpha,
    }
    
    train_loader, val_loader, num_classes = create_data_loaders(
        args.data_dir, args.batch_size, args.num_workers, True, args.model, aug_params
========
    # Automatically enable robust augmentation if model is resnet34_robust
    robust_aug = args.robust or args.model == "resnet34_robust"
    aug_policy = args.aug_policy

    train_loader, val_loader, num_classes = create_data_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        sota_aug=not args.no_sota,
        robust_aug=robust_aug,
        aug_policy=aug_policy,
>>>>>>>> main:experiments/base_ensemble/scripts/train_single_model.py
    )

    # Create model
    pretrained = not args.no_pretrained
    print(
        f"\n{'Loading pretrained' if pretrained else 'Creating'} {args.model} weights..."
    )
    model = create_model(args.model, num_classes, pretrained, device)
    print(f"{args.model} ready with {num_classes} output classes")

    # Training config
    config_dict = {
        "epochs": args.epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_classes": num_classes,
<<<<<<<< HEAD:experiments/resnets_aug/scripts/train.py
        "early_stopping_thresh": args.early_stopping_thresh,
        "top_n_checkpoints": args.top_n_checkpoints,
        "freeze_backbone": args.freeze_backbone,
        "optimizer": args.optimizer,
        "label_smoothing": args.label_smoothing,
        "use_cutmix": args.use_cutmix,
        "cutmix_alpha": args.cutmix_alpha,
========
        "use_sota_aug": not args.no_sota,
        "seed": args.seed,  # Log seed to WandB for reproducibility tracking
        "early_stopping_patience": args.early_stopping_patience,
        "amp": args.amp,
        "grad_clip": args.grad_clip,
        "accum_steps": args.accum_steps,
        "scheduler": args.scheduler,
        "label_smoothing": args.label_smoothing,
        "aug_policy": aug_policy,
>>>>>>>> main:experiments/base_ensemble/scripts/train_single_model.py
    }
    config = TrainingConfig.from_dict(config_dict)

    # Basic summary info is now handled by the TrainerLogger internal to train()
    # but we can keep a high-level model print here
    print(f"   Pretrained: {pretrained}")
<<<<<<<< HEAD:experiments/resnets_aug/scripts/train.py
    print(f"   Save directory: {args.save_dir}")
    print(f"   Wandb logging: {args.wandb}")
    print(f"   Early stopping thresh: {args.early_stopping_thresh}")
    print(f"   Top N checkpoints: {args.top_n_checkpoints}")
    print(f"   Freeze backbone: {args.freeze_backbone}")
========
>>>>>>>> main:experiments/base_ensemble/scripts/train_single_model.py

    # Wandb config
    wandb_config = {
        "project": args.wandb_project,
        "run_name": args.wandb_run_name,
        "entity": getattr(args, "wandb_entity", None),
        "tags": getattr(args, "wandb_tags", None),
        "notes": getattr(args, "wandb_notes", None),
        "mode": getattr(args, "wandb_mode", None),
        "dir": getattr(args, "wandb_dir", None),
        "group": getattr(args, "wandb_group", None),
        "job_type": getattr(args, "wandb_job_type", None),
        # Add augmentation parameters to config for WandB logging
        "augmentation": {
            "num_ops": args.num_ops,
            "magnitude": args.magnitude,
            "use_cutout": args.use_cutout,
            "cutout_size": args.cutout_size,
            "use_mixup": args.use_mixup,
            "mixup_alpha": args.mixup_alpha,
        },
    }

    # Train
    # Include seed and augmentation policy in model name for checkpoint differentiation
    model_save_name = args.model
    if args.aug_policy is not None:
        model_save_name = f"{args.model}_{args.aug_policy}"
    if args.seed is not None:
        model_save_name = f"{model_save_name}_seed{args.seed}"

    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=args.checkpoint_dir,
        model_name=model_save_name,
        wandb_enabled=args.wandb,
        wandb_config=wandb_config,
    )

    return results


if __name__ == "__main__":
    main()
