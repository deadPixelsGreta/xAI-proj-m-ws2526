"""Training and validation routines for classification models."""

import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    epochs: int = 5
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_classes: int = 10
    scheduler: str = "step"
    scheduler_step_size: int = 7
    scheduler_gamma: float = 0.1
    label_smoothing: float = 0.1
    early_stopping_patience: int = 0
    use_sota_aug: bool = True
    amp: bool = False
    grad_clip: float = 0.0
    accum_steps: int = 1
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create a TrainingConfig from a dictionary, ignoring unknown keys."""
        import inspect

        valid_keys = inspect.signature(cls).parameters.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)


class TrainerLogger:
    """Unified logger for console and Weights & Biases."""

    def __init__(
        self,
        enabled: bool = False,
        config: Optional[TrainingConfig] = None,
        model_name: str = "model",
        wandb_config: Optional[Dict] = None,
    ):
        self.enabled = enabled and WANDB_AVAILABLE
        self.model_name = model_name

        if self.enabled:
            init_kwargs = {
                "project": wandb_config.get("project", "imagenet-subset"),
                "name": wandb_config.get("run_name"),
                "entity": wandb_config.get("entity"),
                "tags": wandb_config.get("tags"),
                "notes": wandb_config.get("notes"),
                "mode": wandb_config.get("mode"),
                "dir": wandb_config.get("dir"),
                "group": wandb_config.get("group"),
                "job_type": wandb_config.get("job_type"),
            }
            init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

            wandb.init(
                **init_kwargs,
                config={
                    **(vars(config) if config else {}),
                    "architecture": model_name,
                },
            )

    def watch(self, model: nn.Module):
        if self.enabled:
            wandb.watch(model, log="all", log_freq=100)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.enabled:
            wandb.log(metrics, step=step)

    def update_summary(self, metrics: Dict[str, Any]):
        if self.enabled:
            for k, v in metrics.items():
                wandb.run.summary[k] = v

    def finish(self):
        if self.enabled:
            wandb.finish()

    def info(self, message: str):
        print(message)

    def separator(self, char: str = "=", length: int = 60):
        print(char * length)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 50,
    mixup_cutmix: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    accum_steps: int = 1,
    grad_clip: float = 0.0,
    scheduler: Optional[Any] = None,
) -> Tuple[float, float, float, float]:
    """Train for one epoch and return (loss, accuracy, seconds, grad_norm)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch + 1} [Train]",
    )
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        if mixup_cutmix:
            inputs, targets = mixup_cutmix(inputs, targets)

        # Use AMP if scaler is provided
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Normalize loss for accumulation steps
            loss = loss / accum_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer and update gradients after accumulation
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Compute gradient norm before clipping (for monitoring)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm**0.5

            # Unscale and Clip gradients
            if scaler is not None:
                scaler.unscale_(optimizer)

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # Step scheduler per-batch if it's OneCycleLR
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()

        running_loss += loss.item() * accum_steps

        # For accuracy, use the argmax of the target if it's been mixed (soft labels)
        if mixup_cutmix:
            # If targets are soft labels (from Mixup/Cutmix), we usually don't compute training accuracy the same way
            # or we use the original hard labels for metrics. But for simplicity:
            _, targets_idx = targets.max(1) if targets.ndim > 1 else (None, targets)
        else:
            targets_idx = targets

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets_idx).sum().item()

        # Update progress bar
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc, epoch_time, grad_norm


def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Evaluate on the validation set and return (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    save_dir: str = "checkpoints",
    model_name: str = "model",
    wandb_enabled: bool = False,
    wandb_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run the full training loop and checkpoint the best/final models."""
    # Initialization
    logger = TrainerLogger(wandb_enabled, config, model_name, wandb_config)
    logger.watch(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = _setup_scheduler(optimizer, config, len(train_loader))
    scaler = torch.amp.GradScaler(enabled=config.amp and device.type == "cuda")

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    logger.separator()
    logger.info(f"Starting Training: {model_name}")
    logger.separator()

    mixup_cutmix = _setup_augmentations(config)

    for epoch in range(config.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.epochs}")
        logger.separator("-", 40)

        # Train & Validate
        train_loss, train_acc, epoch_time, grad_norm = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            mixup_cutmix=mixup_cutmix,
            scaler=scaler,
            accum_steps=config.accum_steps,
            grad_clip=config.grad_clip,
            scheduler=scheduler,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if config.scheduler != "onecycle":
            scheduler.step()

        # Logging & Checkpointing
        _log_epoch_summary(
            logger,
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            epoch_time,
            scheduler,
            grad_norm,
            len(train_loader.dataset),
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            _save_checkpoint(
                model,
                optimizer,
                epoch,
                val_acc,
                val_loss,
                config.num_classes,
                model_name,
                save_path / f"best_{model_name}.pth",
            )
            logger.info(f"   New best model saved! (Val Acc: {val_acc:.2f}%)")
            logger.update_summary(
                {"best_val_accuracy": val_acc, "best_epoch": epoch + 1}
            )
        else:
            epochs_without_improvement += 1

        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            logger.info(
                f"\nEarly stopping triggered after {epochs_without_improvement} epochs."
            )
            logger.update_summary(
                {"early_stopped": True, "stopped_at_epoch": epoch + 1}
            )
            break

    # Wrap up
    final_path = save_path / f"final_{model_name}.pth"
    _save_checkpoint(
        model,
        optimizer,
        config.epochs,
        val_acc,
        val_loss,
        config.num_classes,
        model_name,
        final_path,
    )
    logger.update_summary(
        {"final_val_accuracy": val_acc, "final_train_accuracy": train_acc}
    )
    logger.finish()

    logger.separator()
    logger.info("Training Complete!")
    logger.info(
        f"   Best Val Acc: {best_val_acc:.2f}% | Best: {save_path / f'best_{model_name}.pth'}"
    )
    logger.separator()

    return {
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc,
        "best_checkpoint": str(save_path / f"best_{model_name}.pth"),
        "final_checkpoint": str(final_path),
    }


def _setup_scheduler(
    optimizer: optim.Optimizer, config: TrainingConfig, steps_per_epoch: int
):
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.scheduler == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=config.lr,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
        )
    return StepLR(
        optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
    )


def _setup_augmentations(config: TrainingConfig):
    if not config.use_sota_aug:
        return None
    mixup = v2.MixUp(num_classes=config.num_classes, alpha=1.0)
    cutmix = v2.CutMix(num_classes=config.num_classes, alpha=1.0)
    return v2.RandomChoice([mixup, cutmix])


def _save_checkpoint(
    model, optimizer, epoch, val_acc, val_loss, num_classes, model_name, path
):
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "num_classes": num_classes,
        },
        path,
    )


def _log_epoch_summary(
    logger,
    epoch,
    t_loss,
    t_acc,
    v_loss,
    v_acc,
    time,
    scheduler,
    grad_norm,
    dataset_size,
):
    lr = scheduler.get_last_lr()[0]
    logger.info("\n   Epoch Summary:")
    logger.info(f"   Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}%")
    logger.info(f"   Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")
    logger.info(f"   Time: {time:.1f}s | LR: {lr:.6f}")

    logger.log(
        {
            "epoch": epoch + 1,
            "train/loss": t_loss,
            "train/accuracy": t_acc,
            "val/loss": v_loss,
            "val/accuracy": v_acc,
            "learning_rate": lr,
            "epoch_time": time,
            "metrics/generalization_gap": t_acc - v_acc,
            "metrics/grad_norm": grad_norm,
            "metrics/throughput": dataset_size / time,
            "metrics/loss_gap": v_loss - t_loss,
        },
        step=epoch + 1,
    )
