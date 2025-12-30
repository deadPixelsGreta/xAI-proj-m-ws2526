"""Training and validation routines for classification models."""

import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

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
        pbar = tqdm(val_loader, desc="Validating")
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
    config: Dict[str, Any],
    device: torch.device,
    save_dir: str = "checkpoints",
    model_name: str = "model",
    wandb_enabled: bool = False,
    wandb_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run the full training loop and checkpoint the best/final models.

    Config expects keys: epochs, lr, momentum, weight_decay, batch_size, num_classes.
    Returns a summary dict with best/final metrics and checkpoint paths.
    """
    # Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.001),
        momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # Configurable LR scheduler
    scheduler_type = config.get("scheduler", "step").lower()
    epochs = config.get("epochs", 5)

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.get("lr", 0.001),
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )
    else:  # Default to StepLR
        scheduler = StepLR(
            optimizer,
            step_size=config.get("scheduler_step_size", 7),
            gamma=config.get("scheduler_gamma", 0.1),
        )

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Initialize wandb if enabled
    if wandb_enabled and WANDB_AVAILABLE:
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
                **config,
                "architecture": model_name,
            },
        )
        wandb.watch(model, log="all", log_freq=100)

    # AMP Scaler
    scaler = torch.amp.GradScaler(
        enabled=config.get("amp", False) and device.type == "cuda"
    )
    accum_steps = config.get("accum_steps", 1)
    grad_clip = config.get("grad_clip", 0.0)

    # Training loop
    epochs = config.get("epochs", 5)
    best_val_acc = 0.0
    num_classes = config.get("num_classes", 10)

    # Early stopping configuration
    early_stopping_patience = config.get("early_stopping_patience", 0)  # 0 = disabled
    epochs_without_improvement = 0

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    # Setup SOTA batch-level augmentations
    mixup_cutmix = None
    if config.get("use_sota_aug", True):
        mixup = v2.MixUp(num_classes=num_classes, alpha=1.0)
        cutmix = v2.CutMix(num_classes=num_classes, alpha=1.0)
        mixup_cutmix = v2.RandomChoice([mixup, cutmix])

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc, epoch_time, grad_norm = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            mixup_cutmix=mixup_cutmix,
            scaler=scaler,
            accum_steps=accum_steps,
            grad_clip=grad_clip,
            scheduler=scheduler,
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate (OneCycleLR steps per batch inside train_one_epoch if needed,
        # but here we follow standard per-epoch stepping for simplicity unless using OneCycle)
        if scheduler_type != "onecycle":
            scheduler.step()

        # Print epoch summary
        print("\n   Epoch Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Log to wandb
        if wandb_enabled and WANDB_AVAILABLE:
            # Compute additional metrics
            generalization_gap = train_acc - val_acc  # Positive = overfitting
            throughput = len(train_loader.dataset) / epoch_time  # samples/sec

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch_time": epoch_time,
                    # New metrics
                    "metrics/generalization_gap": generalization_gap,
                    "metrics/grad_norm": grad_norm,
                    "metrics/throughput": throughput,
                    "metrics/loss_gap": val_loss - train_loss,
                },
                step=epoch + 1,
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0  # Reset counter
            checkpoint_path = save_path / f"best_{model_name}.pth"
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
                checkpoint_path,
            )
            print(f"   New best model saved! (Val Acc: {val_acc:.2f}%)")

            if wandb_enabled and WANDB_AVAILABLE:
                wandb.run.summary["best_val_accuracy"] = val_acc
                wandb.run.summary["best_epoch"] = epoch + 1
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"\n   Early stopping triggered after {epochs_without_improvement} epochs without improvement."
            )
            if wandb_enabled and WANDB_AVAILABLE:
                wandb.run.summary["early_stopped"] = True
                wandb.run.summary["stopped_at_epoch"] = epoch + 1
            break

    # Save final model
    final_path = save_path / f"final_{model_name}.pth"
    torch.save(
        {
            "epoch": epochs,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "num_classes": num_classes,
        },
        final_path,
    )

    # Finish wandb with explicit summary
    if wandb_enabled and WANDB_AVAILABLE:
        # Set explicit summary metrics to ensure correct values in sweep table
        wandb.run.summary["best_val_accuracy"] = best_val_acc
        wandb.run.summary["final_val_accuracy"] = val_acc
        wandb.run.summary["final_train_accuracy"] = train_acc
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Best model: {save_path / f'best_{model_name}.pth'}")
    print(f"   Final model: {final_path}")

    return {
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc,
        "best_checkpoint": str(save_path / f"best_{model_name}.pth"),
        "final_checkpoint": str(final_path),
    }
