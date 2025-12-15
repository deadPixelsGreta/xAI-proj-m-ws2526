#!/usr/bin/env python3
"""CLI to train supported models on ImageNetSubset-style data."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from experiments.src.utils import get_device, set_seed
from experiments.src.utils.device import get_device_name
from experiments.src.models import SUPPORTED_MODELS, create_model
from experiments.src.data import create_data_loaders, get_dataset_info
from experiments.src.training import train


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

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=SUPPORTED_MODELS,
        help="Model architecture",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch without pretrained weights",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ImageNetSubset",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loader workers"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    # Output arguments
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "checkpoints"),
        help="Directory to save checkpoints",
    )

    # Wandb arguments
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="imagenet-subset", help="W&B project name"
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
            return opt in sys.argv

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
    train_loader, val_loader, num_classes = create_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
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
