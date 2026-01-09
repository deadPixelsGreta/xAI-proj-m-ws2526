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

    def __init__(
        self,
        num_backbones: int,
        num_classes: int,
        feature_dim: int = 3,
        hidden_dims: List[int] = [64, 32],
    ):
        super().__init__()
        # Input: logits from all backbones (num_backbones * num_classes)
        # plus degradation features (feature_dim)
        input_dim = num_backbones * num_classes + feature_dim

        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, num_backbones))

        self.network = nn.Sequential(*layers)

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
        self,
        backbones: List[nn.Module],
        num_classes: int,
        feature_extractor: nn.Module,
        router_hidden_dims: List[int] = [64, 32],
        top_k: int = 0,  # 0 or None means "use all"
    ):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.num_backbones = len(backbones)
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor
        self.top_k = top_k

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

        self.router = MatchingNetwork(
            self.num_backbones, num_classes, hidden_dims=router_hidden_dims
        )

        # Optimization: We no longer freeze backbones here because the
        # backbone_factory handles the specific (partial) freezing logic.

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

        # 5. Apply Top-k if requested (Sparse Weighted Sum)
        # Logic: We keep the weights for top-k models, zero out the rest, and re-normalize?
        # Or just select top-k logits? The standard SEDGE Way:
        # "In inference, top-k selection is applied based on predicted weights."
        if not self.training and self.top_k > 0 and self.top_k < self.num_backbones:
            # Get indices of top-k weights
            topk_vals, topk_indices = torch.topk(weights, self.top_k, dim=1)

            # Create a mask or gather
            # Simpler: just zero out non-topk and re-normalize
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topk_indices, 1.0)

            masked_weights = weights * mask
            # Re-normalize to sum to 1
            masked_weights = masked_weights / (
                masked_weights.sum(dim=1, keepdim=True) + 1e-8
            )

            weights_unsqueezed = masked_weights.unsqueeze(2)
            final_logits = torch.sum(weights_unsqueezed * stacked_logits, dim=1)

        if return_weights:
            return final_logits, weights
        return final_logits
