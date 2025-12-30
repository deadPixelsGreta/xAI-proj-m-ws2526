import torch
import torch.nn as nn
from experiments.base_ensemble.src.inference.ensemble import (
    get_top_predictions,
    ensemble_predict,
)


# Create a mock model
class MockModel(nn.Module):
    def forward(self, x):
        # Return random logits for 10 classes
        return torch.randn(x.size(0), 10)


def test_batch_support():
    print("Testing Batch Support in ensemble.py...")

    models = [MockModel(), MockModel()]
    device = torch.device("cpu")

    # Test batch of 3 images
    batch_size = 3
    num_classes = 10
    image_tensor = torch.randn(batch_size, 3, 224, 224)

    # Test ensemble_predict
    probs, individuals = ensemble_predict(
        models, image_tensor, device, return_individual=True
    )
    print(f"Ensemble Probs Shape: {probs.shape}")  # Should be [3, 10]
    assert probs.shape == (batch_size, num_classes)

    # Test get_top_predictions
    top_k = 3
    class_names = [f"Class{i}" for i in range(num_classes)]
    results = get_top_predictions(probs, top_k=top_k, class_names=class_names)

    print(f"Batch results length: {len(results)}")  # Should be 3
    assert len(results) == batch_size

    for i, res in enumerate(results):
        print(f" Image {i} top-{top_k}: {res}")
        assert len(res) == top_k

    print("\nBatch support verification SUCCESSFUL!")


if __name__ == "__main__":
    test_batch_support()
