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
import numpy as np

from experiments.bagging.src.utils.checkpointing import CheckpointSaver
from torch.amp import autocast, GradScaler

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
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    use_cutmix: bool = False,
    cutmix_alpha: float = 1.0,
    print_freq: int = 50,
    use_amp: bool = True,
) -> Tuple[float, float, float]:
    """Train for one epoch and return (loss, accuracy, seconds)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    scaler = GradScaler('cuda', enabled=use_amp)

    start_time = time.time()

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch + 1} [Train]",
    )
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply CutMix with 50% probability
        if use_cutmix and np.random.rand() < 0.5:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)
            
            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()  
        scaler.step(optimizer)  
        scaler.update()  

        if scheduler is not None:
            scheduler.step()

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
    # Backbone freezing
    freeze_backbone = config.get("freeze_backbone", False)
    opt_models_parameters = model.parameters()
    if freeze_backbone:
        print("Freezing backbone parameters...")
        
        # Determine classifier layer name (ResNet uses 'fc', others might use 'classifier')
        classifier_name = None
        if hasattr(model, 'fc'):
            classifier_name = 'fc'
            opt_models_parameters = model.fc.parameters()
        elif hasattr(model, 'classifier'):
            classifier_name = 'classifier'
            opt_models_parameters = model.classifier.parameters()
        else:
            raise AttributeError("Model has neither 'fc' nor 'classifier' attribute")
        
        # Freeze all parameters except the classifier
        for name, param in model.named_parameters():
            if classifier_name not in name:
                param.requires_grad = False    
        print(f"Backbone frozen. Training only '{classifier_name}' layer.")
    
    # print frozen and unfrozen param, for check, if frozen works
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #opt_params = sum(p.numel() for p in opt_models_parameters) -> error

    print(f"Total parameters: {total_params}")
    print(f"Frozen parameters: {frozen_params}")
    print(f"Unfrozen parameters: {unfrozen_params}")
    #print(f"Optimzer parameters: {opt_params}") -> error
        
    # Setup optimizer based on config
    optimizer_type = config.get("optimizer", "sgd").lower()
    
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(
            opt_models_parameters,
            lr=config.get("lr", 0.001),
            weight_decay=config.get("weight_decay", 1e-4),
            betas=(0.9, 0.999),
        )
    else:  # SGD
        optimizer = optim.SGD(
            opt_models_parameters,
            lr=config.get("lr", 0.001),
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 1e-4),
        )
    
    # Label Smoothing
    label_smoothing = config.get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # tryout OneCycleLR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.get("lr", 0.001) * 3,
        epochs=config.get("epochs", 5),
        steps_per_epoch=len(train_loader),
        pct_start=0.3, 
    )

    # scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

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

        # Merge config with augmentation parameters
        full_config = {
            **config,
            "architecture": model_name,
        }
        
        # Add augmentation parameters if available
        if "augmentation" in wandb_config:
            full_config.update(wandb_config["augmentation"])

        wandb.init(
            **init_kwargs,
            config=full_config,
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
        save_name=wandb_config.get("run_name"),
        wandb_enabled=(wandb_enabled and WANDB_AVAILABLE),  # Pass wandb_enabled flag
        top_n=config.get("top_n_checkpoints", 1),
        early_stop_thresh=config.get("early_stopping_thresh", 5),
    )

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scheduler=scheduler,
            use_cutmix=config.get("use_cutmix", False),
            cutmix_alpha=config.get("cutmix_alpha", 1.0),
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

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
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
