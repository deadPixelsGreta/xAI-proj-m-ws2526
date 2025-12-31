import argparse
import sys
import yaml
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from experiments.base_ensemble.src.data import create_data_loaders
from experiments.base_ensemble.src.utils import get_device
from experiments.SEDGE.models.sedge import SEDGEModel
from experiments.SEDGE.models.backbone_factory import (
    create_all_backbones,
    DEFAULT_BACKBONES,
)
from experiments.SEDGE.data.feature_extractor import ImageFeatureExtractor
from experiments.SEDGE.training.sedge_trainer import SEDGETrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SEDGE model with frozen pretrained backbones"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--data-dir", type=str, default="ImageNetSubset")
    parser.add_argument(
        "--backbones",
        type=str,
        nargs="+",
        default=DEFAULT_BACKBONES,
        help="List of backbone names to use (e.g., resnet34 densenet121 efficientnet_b0 vit_b_16)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiments/SEDGE/checkpoints/sedge.pth",
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


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_params(model: torch.nn.Module) -> int:
    """Count the number of frozen parameters in a model."""
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def main():
    args = parse_args()

    # Load config if provided
    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        # Override args with config values
        tr = cfg.get("training", {})
        if "lr" in tr:
            args.lr = float(tr["lr"])
        if "batch_size" in tr:
            args.batch_size = int(tr["batch_size"])
        if "epochs" in tr:
            args.epochs = int(tr["epochs"])
        if "num_workers" in tr:
            args.num_workers = int(tr["num_workers"])

        dd = cfg.get("data", {})
        if "data_dir" in dd:
            args.data_dir = str(dd["data_dir"])

        md = cfg.get("model", {})
        if "backbones" in md:
            args.backbones = md["backbones"]

    device = get_device()
    num_classes = 10  # ImageNetSubset

    print("=" * 60)
    print("SEDGE Training with Frozen Pretrained Backbones")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Backbones: {args.backbones}")

    # 1. Load frozen pretrained backbones
    print("\nLoading frozen pretrained backbones...")
    backbones, feature_dims = create_all_backbones(args.backbones, num_classes, device)
    print(f"Total backbones loaded: {len(backbones)}")

    # 2. Extract model config
    model_conf = cfg.get("model", {})
    router_dims = model_conf.get("router_hidden_dims", [64, 32])
    top_k = model_conf.get("top_k", 0)

    # 3. Create SEDGE model
    feature_extractor = ImageFeatureExtractor()
    sedge_model = SEDGEModel(
        backbones,
        num_classes,
        feature_extractor,
        router_hidden_dims=router_dims,
        top_k=top_k,
    ).to(device)

    # 4. Print parameter counts
    trainable = count_trainable_params(sedge_model)
    frozen = count_frozen_params(sedge_model)
    print(f"\n📊 Parameter Summary:")
    print(f"   Trainable: {trainable:,} ({trainable / 1e6:.2f}M)")
    print(f"   Frozen:    {frozen:,} ({frozen / 1e6:.2f}M)")
    print(f"   Ratio:     {trainable / (trainable + frozen) * 100:.2f}% trainable")

    # 5. Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # 6. Configure trainer
    trainer_config = cfg.get("training", {})
    trainer_config.update(
        {
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "weight_decay": float(trainer_config.get("weight_decay", 1e-4)),
            "use_group_dro": bool(trainer_config.get("use_group_dro", False)),
            "entropy_reg_weight": float(trainer_config.get("entropy_reg_weight", 0.01)),
            "group_step_size": float(trainer_config.get("group_step_size", 0.01)),
        }
    )

    # 7. Train
    trainer = SEDGETrainer(
        sedge_model, train_loader, val_loader, device, trainer_config
    )
    trainer.fit(args.epochs)

    # 8. Save
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": sedge_model.state_dict(),
            "backbones": args.backbones,
            "num_classes": num_classes,
            "router_hidden_dims": router_dims,
            "top_k": top_k,
        },
        args.save_path,
    )
    print(f"\n✅ SEDGE model saved to {args.save_path}")


if __name__ == "__main__":
    main()
