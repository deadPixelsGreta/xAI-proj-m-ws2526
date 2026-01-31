import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibratedModel(nn.Module):
    """
    A wrapper for any model to apply Temperature Scaling.
    The temperature 'T' is a single scalar that softens/sharpens the softmax.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Initialize temperature at 1.0 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Divide logits by temperature.
        T > 1: Softens distribution (reduces overconfidence)
        T < 1: Sharpens distribution (increases confidence)
        """
        # Expand temperature to match batch size
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
