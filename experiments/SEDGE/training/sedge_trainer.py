import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict
from tqdm import tqdm


class SEDGETrainer:
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

        self.criterion = nn.CrossEntropyLoss(reduction="none")  # None for GroupDRO
        self.entropy_reg_weight = config.get("entropy_reg_weight", 0.01)
        self.use_group_dro = config.get("use_group_dro", False)

        # For GroupDRO: keep track of group losses and weights
        # If enabled, we need to know the number of groups.
        # Ideally this comes from the dataset or config.
        # Fallback to a safe default if not specified, or disable.
        # User config has list of corruptions, length of that list is num_groups.
        # But for now, let's play safe and check if explicit group count is given.
        self.num_groups = config.get(
            "num_groups", 5
        )  # Default to 5 (Clean + 4 corruptions or just 5)

        self.group_counts = torch.ones(self.num_groups, device=device)
        self.group_weights = torch.ones(self.num_groups, device=device) / float(
            self.num_groups
        )
        self.group_step_size = config.get("group_step_size", 0.01)

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
        print(f"Training SEDGE for {epochs} epochs...")
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()
            print(
                f"Epoch {epoch + 1}: Train Loss {train_loss:.4f} Acc {train_acc:.2f} | Val Loss {val_loss:.4f} Acc {val_acc:.2f}"
            )
