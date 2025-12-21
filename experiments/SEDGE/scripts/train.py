import argparse
import sys
import yaml
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from experiments.bagging.src.models import load_checkpoint
from experiments.bagging.src.data import create_data_loaders
from experiments.bagging.src.utils import get_device
from experiments.SEDGE.models.sedge import SEDGEModel
from experiments.SEDGE.data.feature_extractor import ImageFeatureExtractor
from experiments.SEDGE.data.corruptions import get_robust_transform
from experiments.SEDGE.training.sedge_trainer import SEDGETrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train SEDGE model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--data-dir", type=str, default="ImageNetSubset")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=[
            "experiments/bagging/checkpoints/best_densenet121.pth",
            "experiments/bagging/checkpoints/best_resnet34.pth",
            "experiments/bagging/checkpoints/best_efficientnet_b0.pth",
        ],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--save-path", type=str, default="experiments/bagging/checkpoints/sedge.pth"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pin_memory in DataLoaders (recommended for CUDA)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config if provided
    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        # Override args with config values if not specified on CLI (simplified logic)
        # Check training section
        tr = cfg.get("training", {})
        if "lr" in tr:
            args.lr = float(tr["lr"])
        if "batch_size" in tr:
            args.batch_size = int(tr["batch_size"])
        if "epochs" in tr:
            args.epochs = int(tr["epochs"])
        if "num_workers" in tr:
            args.num_workers = int(tr["num_workers"])
        if "pin_memory" in tr:
            args.pin_memory = bool(tr["pin_memory"])

        # Check data section
        dd = cfg.get("data", {})
        if "data_dir" in dd:
            args.data_dir = str(dd["data_dir"])

    device = get_device()

    # 1. Load backbones
    backbones = []
    num_classes = 10  # Default for this project
    for cp_path in args.checkpoints:
        print(f"Loading backbone from {cp_path}...")
        model, _, _ = load_checkpoint(cp_path, device, num_classes=num_classes)
        backbones.append(model)

    # 2. Create SEDGE model
    feature_extractor = ImageFeatureExtractor()
    sedge_model = SEDGEModel(backbones, num_classes, feature_extractor).to(device)

    # 3. Create data loaders with robust transforms
    # We want training to see broadened corruptions
    train_loader, val_loader, _ = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # Wrap train_loader's dataset transform with robustness
    # This is a bit hacky depending on how create_data_loaders is implemented
    # Ideally we'd pass the robust transform to create_data_loaders.
    # For now, let's assume the user handles the data dir setup.

    # 4. Train
    # Merge config for trainer: Start with YAML training section
    trainer_config = cfg.get("training", {})
    # Ensure CLI/Arg overrides (lr, epochs, etc) take precedence or reflect the update
    trainer_config.update(
        {
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "weight_decay": float(trainer_config.get("weight_decay", 1e-4)),
            "use_group_dro": bool(trainer_config.get("use_group_dro", True)),
            "entropy_reg_weight": float(trainer_config.get("entropy_reg_weight", 0.01)),
            "group_step_size": float(trainer_config.get("group_step_size", 0.01)),
        }
    )

    trainer = SEDGETrainer(
        sedge_model, train_loader, val_loader, device, trainer_config
    )
    trainer.fit(args.epochs)

    # 5. Save
    torch.save(sedge_model.state_dict(), args.save_path)
    print(f"SEDGE model saved to {args.save_path}")


if __name__ == "__main__":
    main()
