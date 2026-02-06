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

    # Fallback: use parent's parent (scripts -> experiments -> project root)
    project_root = (current.parent.parent).resolve()
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

from experiments.resnets_aug.src.utils import get_device, set_seed
from experiments.resnets_aug.src.utils.device import get_device_name
from experiments.resnets_aug.src.models import SUPPORTED_MODELS, create_model
from experiments.resnets_aug.src.data import create_data_loaders, get_dataset_info
from experiments.resnets_aug.src.training import train


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
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "checkpoints"),
        help="Directory to save checkpoints",
    )

    # Wandb arguments
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

        def cli_overrides(opt: str) -> bool:
            return any(arg == opt or arg.startswith(f"{opt}=") for arg in sys.argv)

        # training section
        tr = cfg.get("training", {})
        if "epochs" in tr and not cli_overrides("--epochs"):
            args.epochs = int(tr["epochs"])
        if "batch_size" in tr and not cli_overrides("--batch-size"):
            args.batch_size = int(tr["batch_size"])
        if "learning_rate" in tr and not cli_overrides("--lr"):
            args.lr = float(tr["learning_rate"])
        if "momentum" in tr and not cli_overrides("--momentum"):
            args.momentum = float(tr["momentum"])
        if "weight_decay" in tr and not cli_overrides("--weight-decay"):
            args.weight_decay = float(tr["weight_decay"])
        if "num_workers" in tr and not cli_overrides("--num-workers"):
            args.num_workers = int(tr["num_workers"])
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
        if "pretrained" in md and not cli_overrides("--no-pretrained"):
            # CLI flag disables pretrained; YAML true => keep pretrained
            pretrained_yaml = bool(md["pretrained"])
            args.no_pretrained = not pretrained_yaml

        # data section
        dd = cfg.get("data", {})
        if "data_dir" in dd and not cli_overrides("--data-dir"):
            args.data_dir = str(dd["data_dir"])

        # paths section
        pd = cfg.get("paths", {})
        if "checkpoint_dir" in pd and not cli_overrides("--save-dir"):
            args.save_dir = str(pd["checkpoint_dir"])

        # wandb section
        wb = cfg.get("wandb", {})
        if "enabled" in wb and not cli_overrides("--wandb"):
            args.wandb = bool(wb["enabled"])
        if "project" in wb and not cli_overrides("--wandb-project"):
            args.wandb_project = str(wb["project"])
        # Optional W&B fields from YAML (no CLI flags defined for these)
        if "run_name" in wb:
            args.wandb_run_name = wb.get("run_name")
        if "entity" in wb:
            args.wandb_entity = wb.get("entity")
        if "tags" in wb:
            args.wandb_tags = wb.get("tags")
        if "notes" in wb:
            args.wandb_notes = wb.get("notes")
        if "mode" in wb:
            args.wandb_mode = wb.get("mode")
        if "dir" in wb:
            args.wandb_dir = wb.get("dir")
        if "group" in wb:
            args.wandb_group = wb.get("group")
        if "job_type" in wb:
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
    print(f"\n Dataset Summary:")
    print(f"   Training samples: {dataset_info['train_samples']}")
    print(f"   Validation samples: {dataset_info['val_samples']}")
    print(f"   Classes: {dataset_info['classes']}")
    print(f"   Number of classes: {dataset_info['num_classes']}")

    # Create data loaders
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
    )

    # Create model
    pretrained = not args.no_pretrained
    print(
        f"\n{'Loading pretrained' if pretrained else 'Creating'} {args.model} weights..."
    )
    model = create_model(args.model, num_classes, pretrained, device)
    print(f"{args.model} ready with {num_classes} output classes")

    # Training config
    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_classes": num_classes,
        "early_stopping_thresh": args.early_stopping_thresh,
        "top_n_checkpoints": args.top_n_checkpoints,
        "freeze_backbone": args.freeze_backbone,
        "optimizer": args.optimizer,
        "label_smoothing": args.label_smoothing,
        "use_cutmix": args.use_cutmix,
        "cutmix_alpha": args.cutmix_alpha,
    }

    print(f"\n Training Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Momentum: {args.momentum}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Pretrained: {pretrained}")
    print(f"   Save directory: {args.save_dir}")
    print(f"   Wandb logging: {args.wandb}")
    print(f"   Early stopping thresh: {args.early_stopping_thresh}")
    print(f"   Top N checkpoints: {args.top_n_checkpoints}")
    print(f"   Freeze backbone: {args.freeze_backbone}")

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
    # Include seed in model name for checkpoint differentiation
    model_save_name = args.model
    if args.seed is not None:
        model_save_name = f"{args.model}_seed{args.seed}"

    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=args.save_dir,
        model_name=model_save_name,
        wandb_enabled=args.wandb,
        wandb_config=wandb_config,
    )

    return results


if __name__ == "__main__":
    main()
