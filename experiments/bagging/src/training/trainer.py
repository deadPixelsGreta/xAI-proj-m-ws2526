"""Training and validation routines for classification models."""

import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.bagging.src.utils.checkpointing import CheckpointSaver

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
) -> Tuple[float, float, float]:
    """Train for one epoch and return (loss, accuracy, seconds)."""
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

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc, epoch_time


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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.001),
        momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

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

    # Training loop
    epochs = config.get("epochs", 5)
    best_val_acc = 0.0
    num_classes = config.get("num_classes", 10)

     # Initialize checkpoint saver
    checkpoint_saver = CheckpointSaver(
        dirpath=save_path,
        num_classes=num_classes,
        model_name=model_name,
        wandb_enabled=(wandb_enabled and WANDB_AVAILABLE),  # Pass wandb_enabled flag
    )

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Print epoch summary
        print(f"\n   Epoch Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Log to wandb
        if wandb_enabled and WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch_time": epoch_time,
                },
                step=epoch + 1,  # Explicitly set step for proper x-axis in charts
            )

        # Save best model new way with Early Stopping
        early_stop = checkpoint_saver(model, epoch, val_acc, val_loss, optimizer)
        if early_stop:
            print("Early stopping triggered. Stopping training.")
            break
        
        # Save best model old way
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     checkpoint_path = save_path / f"best_{model_name}.pth"
        #     torch.save(
        #         {
        #             "epoch": epoch,
        #             "model_name": model_name,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "val_acc": val_acc,
        #             "val_loss": val_loss,
        #             "num_classes": num_classes,
        #         },
        #         checkpoint_path,
        #     )
        #     print(f"   New best model saved! (Val Acc: {val_acc:.2f}%)")

        #     if wandb_enabled and WANDB_AVAILABLE:
        #         wandb.run.summary["best_val_accuracy"] = val_acc
        #         wandb.run.summary["best_epoch"] = epoch + 1

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
