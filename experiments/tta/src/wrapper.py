import torch
import torch.nn as nn
from torchvision import transforms
from typing import Tuple, List

class TTAWrapper(nn.Module):
    """
    Wraps a model to perform Test-Time Augmentation (TTA).
    Produces multiple augmented versions of each image and averages the resulting probabilities.
    """
    def __init__(self, model: nn.Module, num_samples: int = 5):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        
        # TTA transforms applied to already-normalized tensors
        # Focus on stable transforms for domain-shifted data
        self.tta_transforms = nn.ModuleList([
            transforms.RandomHorizontalFlip(p=1.0), # Force flip for specific pass
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input batch [B, C, H, W]
        Returns:
            avg_logits: Averaged logits across TTA samples [B, NumClasses]
            avg_weights: Averaged routing weights if available [B, NumExperts]
        """
        all_probs = []
        all_weights = []
        
        # 1. First pass: Original image
        logits, weights = self.model(x)
        all_probs.append(torch.softmax(logits, dim=1))
        if weights is not None:
            all_weights.append(weights)
            
        # 2. Second pass: Flipped image
        flipped_x = torch.flip(x, dims=[-1])
        logits, weights = self.model(flipped_x)
        all_probs.append(torch.softmax(logits, dim=1))
        if weights is not None:
            all_weights.append(weights)
        
        # 3. Average the probabilities
        stacked_probs = torch.stack(all_probs, dim=0)
        avg_probs = stacked_probs.mean(dim=0)
        avg_logits = torch.log(avg_probs + 1e-9)
        
        avg_weights = None
        if all_weights:
            avg_weights = torch.stack(all_weights, dim=0).mean(dim=0)
            
        return avg_logits, avg_weights

