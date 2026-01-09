import argparse
import sys
import yaml
import torch
import wandb
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
from experiments.SEDGE.training.console_ui import (
    ConsoleUI,
    Colors,
    color,
    format_params,
    format_number,
)


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
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to directory with fine-tuned checkpoints (e.g., experiments/base_ensemble/checkpoints). If not provided, uses ImageNet pretrained weights.",
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

    # Initialize wandb
    # If run in a sweep, this will pick up the run ID and config
    wandb.init(project="SEDGE-Training")

    # Enable TF32 for Ampere GPUs (like RTX A4000)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
        # Load checkpoint_dir from config if not specified via CLI
        if args.checkpoint_dir is None and "backbone_checkpoint_dir" in md:
            checkpoint_dir = md["backbone_checkpoint_dir"]
            if checkpoint_dir:  # Only set if not null/None
                args.checkpoint_dir = str(checkpoint_dir)

    # 3. Override with WandB sweep parameters if available
    # This is crucial for sweeps to work!
    if wandb.config:
        # Map flat wandb.config to our nested structure or args
        if "lr" in wandb.config:
            args.lr = wandb.config.lr
        if "batch_size" in wandb.config:
            args.batch_size = wandb.config.batch_size
        if "epochs" in wandb.config:
            args.epochs = wandb.config.epochs
        
        # Merge other wandb.config into cfg for the trainer
        for key, value in wandb.config.items():
            if key in ["lr", "batch_size", "epochs"]:
                continue
            if "training" not in cfg:
                cfg["training"] = {}
            cfg["training"][key] = value

    device = get_device()
    num_classes = 10  # ImageNetSubset

    # ═══════════════════════════════════════════════════════════════════════
    # Beautiful Header
    # ═══════════════════════════════════════════════════════════════════════
    ConsoleUI.header(
        "SEDGE Training",
        subtitle="Sparse Expert Diverse Generalization Ensemble with Frozen Backbones",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Environment Info
    # ═══════════════════════════════════════════════════════════════════════
    ConsoleUI.section("Environment")

    device_name = str(device).upper()
    if "cuda" in str(device):
        try:
            gpu_name = torch.cuda.get_device_name(0)
            device_name = f"CUDA ({gpu_name})"
        except Exception:
            device_name = "CUDA"

    ConsoleUI.key_value("Device", color(device_name, Colors.GREEN))
    ConsoleUI.key_value("PyTorch Version", torch.__version__)

    # ═══════════════════════════════════════════════════════════════════════
    # Configuration Summary
    # ═══════════════════════════════════════════════════════════════════════
    ConsoleUI.section("Configuration")
    ConsoleUI.key_value("Data Directory", args.data_dir)
    ConsoleUI.key_value("Batch Size", args.batch_size)
    ConsoleUI.key_value("Learning Rate", f"{args.lr:.2e}")
    ConsoleUI.key_value("Epochs", args.epochs)
    ConsoleUI.key_value("Num Workers", args.num_workers)
    if args.checkpoint_dir:
        ConsoleUI.key_value("Checkpoint Dir", color(args.checkpoint_dir, Colors.GREEN))
    else:
        ConsoleUI.key_value(
            "Backbone Weights", color("ImageNet Pretrained", Colors.YELLOW)
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Load Backbones
    # ═══════════════════════════════════════════════════════════════════════
    ConsoleUI.section("Loading Backbones")
    ConsoleUI.info(f"Backbones to load: {color(str(len(args.backbones)), Colors.BOLD)}")

    for i, name in enumerate(args.backbones, 1):
        ConsoleUI.info(f"  {i}. {color(name, Colors.CYAN)}")

    print()  # Spacing before downloads

    backbones, feature_dims = create_all_backbones(
        args.backbones, num_classes, device, checkpoint_dir=args.checkpoint_dir
    )

    ConsoleUI.success(f"All {len(backbones)} backbones loaded successfully!")

    # ═══════════════════════════════════════════════════════════════════════
    # Build Model
    # ═══════════════════════════════════════════════════════════════════════
    ConsoleUI.section("Model Architecture")

    # Extract model config
    model_conf = cfg.get("model", {})
    router_dims = model_conf.get("router_hidden_dims", [64, 32])
    top_k = model_conf.get("top_k", 0)

    # Override with sweep parameters if present
    if wandb.config:
        if "router_hidden_dim_1" in wandb.config and "router_hidden_dim_2" in wandb.config:
            router_dims = [wandb.config.router_hidden_dim_1, wandb.config.router_hidden_dim_2]
        if "top_k" in wandb.config:
            top_k = wandb.config.top_k

    # Create SEDGE model
    feature_extractor = ImageFeatureExtractor()
    sedge_model = SEDGEModel(
        backbones,
        num_classes,
        feature_extractor,
        router_hidden_dims=router_dims,
        top_k=top_k,
    ).to(device)

    # Parameter counts
    trainable = count_trainable_params(sedge_model)
    frozen = count_frozen_params(sedge_model)
    total = trainable + frozen
    ratio = trainable / total * 100 if total > 0 else 0

    ConsoleUI.stats_box(
        "Parameter Summary",
        {
            "Trainable Parameters": f"{format_number(trainable)} ({format_params(trainable)})",
            "Frozen Parameters": f"{format_number(frozen)} ({format_params(frozen)})",
            "Total Parameters": f"{format_number(total)} ({format_params(total)})",
            "Trainable Ratio": f"{ratio:.4f}%",
        },
    )

    ConsoleUI.key_value("Router Hidden Dims", str(router_dims))
    ConsoleUI.key_value(
        "Top-K Selection", str(top_k) if top_k > 0 else "Disabled (use all)"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Create Data Loaders
    # ═══════════════════════════════════════════════════════════════════════
    ConsoleUI.section("Data Loading")

    train_loader, val_loader, _ = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    ConsoleUI.key_value("Training Batches", len(train_loader))
    ConsoleUI.key_value("Validation Batches", len(val_loader))
    ConsoleUI.key_value("Training Samples", len(train_loader.dataset))
    ConsoleUI.key_value("Validation Samples", len(val_loader.dataset))

    # ═══════════════════════════════════════════════════════════════════════
    # Configure Trainer
    # ═══════════════════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════════════════
    # Train
    # ═══════════════════════════════════════════════════════════════════════
    trainer = SEDGETrainer(
        sedge_model, train_loader, val_loader, device, trainer_config
    )
    best_acc, total_time, epochs_completed = trainer.fit(args.epochs)

    # ═══════════════════════════════════════════════════════════════════════
    # Save Model
    # ═══════════════════════════════════════════════════════════════════════
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

    ConsoleUI.training_complete(
        best_acc=best_acc,
        total_time=total_time,
        total_epochs=epochs_completed,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
