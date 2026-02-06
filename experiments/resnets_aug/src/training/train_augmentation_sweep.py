"""Training script for WandB augmentation parameter sweeps."""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import wandb
from torchvision import models
from tqdm import tqdm


# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import create_data_loaders


def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    """Load a pretrained model and adapt for num_classes."""
    model_dict = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
    }
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model_dict[model_name](pretrained=pretrained)
    
    # Adapt final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

# 
def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=True):
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=True):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc="Validation", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
            
        with autocast(enabled=use_amp):    
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def extract_aug_params(config, model_name):
    """Extract augmentation parameters based on model type."""
    aug_params = {}
    
    if model_name == "resnet18":
        aug_params = {
            'brightness': config.get('brightness', 0.3),
            'contrast': config.get('contrast', 0.3),
            'saturation': config.get('saturation', 0.3),
            'hue': config.get('hue', 0.1),
            'rotation': config.get('rotation', 15),
        }
    elif model_name == "resnet34":
        aug_params = {
            'policy': config.get('policy', 'IMAGENET'),
            'use_trivial_augment': config.get('use_trivial_augment', False),
        }
    elif model_name == "resnet50":
        aug_params = {
            'num_ops': config.get('num_ops', 2),
            'magnitude': config.get('magnitude', 9),
            'use_cutout': config.get('use_cutout', True),
        }
    
    return aug_params

def train():
    """Main training function for sweep."""
    # Initialize wandb
    wandb.init()
    config = wandb.config
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Extract augmentation parameters based on model
    aug_params = extract_aug_params(config, config.model_name)
    model_name = config.model_name

    print(f"\nModel: {model_name}")
    print("Augmentation parameters:")
    for k, v in aug_params.items():
        print(f"  {k}: {v}")
    
    # Create data loaders with augmentation parameters
    data_dir = os.environ.get('DATA_DIR', 'data/imagenet-subset')
    train_loader, val_loader, num_classes = create_data_loaders(
        data_dir=data_dir,
        batch_size=config.batch_size,
        model_name=model_name,
        aug_params=aug_params,
        num_workers=4,
        pin_memory=True
    )

    print(f"Classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model = get_model(model_name, num_classes, pretrained=True)
    model = model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.get('weight_decay', 0.01),)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, epochs=config.epochs, steps_per_epoch=len(train_loader),)
    scaler = GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs}")
        print('='*60)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wandb.run.summary['best_val_accuracy'] = best_val_acc
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    wandb.finish()


if __name__ == "__main__":
    train()