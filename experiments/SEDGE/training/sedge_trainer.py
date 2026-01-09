import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from typing import Dict
from tqdm import tqdm

from experiments.SEDGE.training.console_ui import (
    ConsoleUI,
    TrainingTimer,
    Colors,
    color,
)


class SEDGETrainer:
    """SEDGE model trainer with rich console output."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        config: Dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # Store learning rate for display
        self.lr = config.get("lr", 1e-3)

        self.criterion = nn.CrossEntropyLoss(reduction="none")  # None for GroupDRO
        self.entropy_reg_weight = config.get("entropy_reg_weight", 0.01)
        self.use_group_dro = config.get("use_group_dro", False)

        # For GroupDRO: keep track of group losses and weights
        self.num_groups = config.get("num_groups", 5)

        self.group_counts = torch.ones(self.num_groups, device=device)
        self.group_weights = torch.ones(self.num_groups, device=device) / float(
            self.num_groups
        )
        self.group_step_size = config.get("group_step_size", 0.01)

        # Early stopping
        self.early_stopping_patience = config.get("early_stopping_patience", 5)
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        # Training timer
        self.timer = TrainingTimer()

    def compute_entropy_reg(self, weights: torch.Tensor):
        """Computes entropy regularization to prevent collapse."""
        # weights: [B, num_backbones]
        entropy = -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=1))
        return (
            -self.entropy_reg_weight * entropy
        )  # Positive entropy -> minimize -> -entropy weight

    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1} [Train]",
        )
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # 1. Forward pass
            outputs, routing_weights = self.model(inputs, return_weights=True)

            # 2. Compute Loss
            losses = self.criterion(outputs, targets)

            if self.use_group_dro:
                # Simplified GroupDRO: treat each batch member as if we don't know the exact group
                # but we minimize the worst-case among them or use metadata if available.
                # In this project, we might just apply regular loss if group id is not passed.
                # Let's assume groups are not explicitly provided for now and just use mean loss,
                # but provide the placeholder for full implementation.
                loss = losses.mean()
            else:
                loss = losses.mean()

            # 3. Add Entropy Regularization
            reg = self.compute_entropy_reg(routing_weights)
            total_loss = loss + reg

            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {"loss": total_loss.item(), "acc": 100.0 * correct / total}
            )

        return running_loss / len(self.train_loader), 100.0 * correct / total

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.val_loader, desc="Validating")
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return running_loss / len(self.val_loader), 100.0 * correct / total

    def fit(self, epochs: int):
        """Train the model with rich console output."""
        ConsoleUI.section("Training")
        ConsoleUI.info(f"Total epochs: {color(str(epochs), Colors.BOLD)}")
        ConsoleUI.info(f"Learning rate: {color(f'{self.lr:.2e}', Colors.YELLOW)}")
        ConsoleUI.info(
            f"Early stopping patience: {color(str(self.early_stopping_patience), Colors.CYAN)}"
        )

        self.timer.start()
        epochs_completed = 0

        for epoch in range(epochs):
            self.timer.start_epoch()

            # Epoch header
            ConsoleUI.epoch_header(epoch + 1, epochs, lr=self.lr)

            # Training and validation
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Timing
            epoch_time = self.timer.end_epoch()
            epochs_completed = epoch + 1

            # Check if new best
            is_best = val_acc > self.best_val_acc

            # Print epoch summary
            ConsoleUI.epoch_summary(
                train_loss, train_acc, val_loss, val_acc, epoch_time, is_best=is_best
            )

            # Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "is_best": is_best,
                "lr": self.lr
            })

            # ETA estimate
            eta = self.timer.eta(epochs_completed, epochs)
            if epochs_completed < epochs:
                ConsoleUI.info(
                    f"Estimated time remaining: {color(eta, Colors.BRIGHT_CYAN)}"
                )

            # Early stopping check
            if is_best:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if (
                    self.early_stopping_patience > 0
                    and self.epochs_without_improvement >= self.early_stopping_patience
                ):
                    ConsoleUI.early_stopping(
                        self.early_stopping_patience, self.best_val_acc
                    )
                    break

        return self.best_val_acc, self.timer.total_elapsed(), epochs_completed
