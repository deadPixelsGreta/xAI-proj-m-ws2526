import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class MoERouter(nn.Module):
    """
    Lightweight MLP Router that takes features from a backbone
    and predicts weights for each expert in the ensemble.
    """
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        # Softmax ensures weights sum to 1.0 across experts
        # Using a small temperature can make the distribution sharper if needed
        logits = self.net(x)
        return F.softmax(logits, dim=1)

class FeatureWrapper(nn.Module):
    """
    Wraps a torchvision model to return both (logits, features).
    Handles different architectures explicitly.
    """
    def __init__(self, model: nn.Module, model_name: str):
        super().__init__()
        self.model = model
        self.model_name = model_name.lower()
        
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if "resnet" in self.model_name:
            # ResNet: features are after global pooling
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            features = torch.flatten(x, 1)
            logits = self.model.fc(features)
            return logits, features
            
        elif "densenet" in self.model_name:
            # DenseNet: features are from self.model.features
            features_raw = self.model.features(x)
            out = F.relu(features_raw, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            features = torch.flatten(out, 1)
            logits = self.model.classifier(features)
            return logits, features
            
        elif "efficientnet" in self.model_name:
            # EfficientNet: features from self.model.features + self.model.avgpool
            features_raw = self.model.features(x)
            out = self.model.avgpool(features_raw)
            features = torch.flatten(out, 1)
            logits = self.model.classifier(features)
            return logits, features
            
        elif "vit" in self.model_name:
            # ViT: features before heads
            # _process_input handles patch embedding
            x = self.model._process_input(x)
            n = x.shape[0]
            # Expand class token
            cls_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.model.encoder(x)
            # Classifier "token" is first
            features = x[:, 0]
            logits = self.model.heads(features)
            return logits, features
            
        else:
            # Generic fallback (might not work for all)
            logits = self.model(x)
            return logits, logits # Can't easily extract features without knowledge

class MoEEnsemble(nn.Module):
    """
    Full Mixture-of-Experts Ensemble.
    Combines N experts using a dynamic Router.
    """
    def __init__(self, experts: List[FeatureWrapper], router: MoERouter, feature_provider_idx: int = 3, temperatures: List[float] = None):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.feature_provider_idx = feature_provider_idx # Which expert provides features to the router
        
        if temperatures is None:
            self.temperatures = [1.0] * len(experts)
        else:
            self.temperatures = temperatures
        # Register as buffer so it moves with the model to device
        self.register_buffer("temps", torch.tensor(self.temperatures))

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_logits = []
        router_features = None
        
        # 1. Get predictions from all experts
        for i, expert in enumerate(self.experts):
            logits, feats = expert(x)
            # Apply temperature scaling: logits / T
            scaled_logits = logits / self.temps[i]
            expert_logits.append(scaled_logits)
            if i == self.feature_provider_idx:
                router_features = feats
        
        # 2. Get routing weights
        # We detach features to ensure we don't backprop into experts if they aren't frozen
        # But for MoE training we usually freeze them anyway.
        routing_weights = self.router(router_features.detach()) # [Batch, NumExperts]
        
        # 3. Combine logits: [Batch, NumExperts, NumClasses]
        stacked_logits = torch.stack(expert_logits, dim=1)
        
        # weighted_logits = sum(w_i * logits_i)
        # Using unsqueeze for broadcasting: [B, E, 1] * [B, E, C] -> [B, E, C] -> sum over E -> [B, C]
        weighted_logits = (stacked_logits * routing_weights.unsqueeze(-1)).sum(dim=1)
        
        return weighted_logits, routing_weights
