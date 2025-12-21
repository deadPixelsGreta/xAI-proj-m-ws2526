import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class LabelSpaceAdapter(nn.Module):
    """Linear adapter to map backbone features/logits to target label space."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.adapter = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.adapter(x)


class MatchingNetwork(nn.Module):
    """Router that predicts ensemble weights for backbones."""

    def __init__(self, num_backbones: int, num_classes: int, feature_dim: int = 3):
        super().__init__()
        # Input: logits from all backbones (num_backbones * num_classes)
        # plus degradation features (feature_dim)
        input_dim = num_backbones * num_classes + feature_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_backbones),
        )

    def forward(
        self, logits_list: List[torch.Tensor], degradation_features: torch.Tensor
    ):
        # Concatenate all logits
        combined_logits = torch.cat(
            logits_list, dim=1
        )  # [B, num_backbones * num_classes]

        # Concatenate with degradation features
        x = torch.cat([combined_logits, degradation_features], dim=1)

        # Predict weights and normalize via Softmax
        weights = self.network(x)
        return F.softmax(weights, dim=1)


class SEDGEModel(nn.Module):
    """Specialized Ensemble Learning for Domain Generalization."""

    def __init__(
        self, backbones: List[nn.Module], num_classes: int, feature_extractor: nn.Module
    ):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.num_backbones = len(backbones)
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor

        # In SEDGE, we assume backbones are fixed and we adapt their outputs
        # Each backbone has its own LabelSpaceAdapter
        # For simplicity, we assume backbones output logits of size 'num_classes'
        # If they don't, we'd need to know their feature dim.
        self.adapters = nn.ModuleList(
            [
                LabelSpaceAdapter(num_classes, num_classes)
                for _ in range(self.num_backbones)
            ]
        )

        self.router = MatchingNetwork(self.num_backbones, num_classes)

        # Freeze backbones
        for bb in self.backbones:
            for param in bb.parameters():
                param.requires_grad = False

    def forward(self, x, return_weights=False):
        # 1. Extract degradation features
        deg_features = self.feature_extractor(x)

        # 2. Get backbone outputs and adapt them
        backbone_logits = []
        adapted_logits = []
        for i, bb in enumerate(self.backbones):
            with torch.no_grad():
                logits = bb(x)
            backbone_logits.append(logits)
            adapted_logits.append(self.adapters[i](logits))

        # 3. Get routing weights
        weights = self.router(backbone_logits, deg_features)  # [B, num_backbones]

        # 4. Ensemble: sum(w_i * adapted_logits_i)
        # weights: [B, num_backbones] -> [B, num_backbones, 1]
        # adapted_logits: list of [B, num_classes] -> [B, num_backbones, num_classes]
        weights_unsqueezed = weights.unsqueeze(2)
        stacked_logits = torch.stack(adapted_logits, dim=1)

        final_logits = torch.sum(weights_unsqueezed * stacked_logits, dim=1)

        if return_weights:
            return final_logits, weights
        return final_logits
