"""A100-optimized trainer with mixed precision and advanced features."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
from tqdm import tqdm
import wandb


class A100Trainer:
    """Trainer optimized for A100 GPU with mixed precision training."""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        device: torch.device,
        use_amp: bool = True,
        use_compile: bool = True,
    ):
        """
        Initialize A100-optimized trainer.
        
        Args:
            model_name: resnet18, resnet34, or resnet50
            num_classes: Number of output classes
            device: Training device (cuda)
            use_amp: Use automatic mixed precision
            use_compile: Use torch.compile for speedup (PyTorch 2.0+)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.use_amp = use_amp
        
        # Create model
        self.model = self._create_model(model_name, num_classes)
        self.model = self.model.to(device)
        
        # Compile model if available (PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            print("✓ Model compiled with torch.compile()")
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=use_amp)
        
        # Training state
        self.best_val_acc = 0.0
        self.current_epoch = 0
    
    def _create_model(self, model_name: str, num_classes: int) -> nn.Module:
        """Create pretrained model."""
        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
        }
        
        if model_name not in model_dict:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model_dict[model_name](pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        return model
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> tuple:
        """Train for one epoch with mixed precision."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient
            
            # Mixed precision forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> tuple:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, filepath: str, optimizer: optim.Optimizer, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'scaler_state_dict': self.scaler.state_dict(),
            **kwargs
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str, optimizer: optim.Optimizer = None):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.scaler.load_state_dict(checkpoint.get('scaler_state_dict', {}))
        
        return checkpoint


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Create AdamW optimizer with config."""
    return optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 0.01),
        betas=(0.9, 0.999),
    )


def create_scheduler(
    optimizer: optim.Optimizer, 
    config: dict, 
    steps_per_epoch: int
) -> optim.lr_scheduler._LRScheduler:
    """Create OneCycleLR scheduler."""
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.get('learning_rate', 0.001),
        epochs=config.get('epochs', 30),
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0,
    )


def create_criterion(config: dict) -> nn.Module:
    """Create loss criterion with label smoothing."""
    return nn.CrossEntropyLoss(
        label_smoothing=config.get('label_smoothing', 0.1)
    )