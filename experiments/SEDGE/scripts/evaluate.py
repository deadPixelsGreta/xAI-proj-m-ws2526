import torch
import torch.nn.functional as F
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from experiments.base_ensemble.src.data import create_data_loaders, create_test_loader
from experiments.base_ensemble.src.utils import get_device
from experiments.SEDGE.models.sedge import SEDGEModel
from experiments.SEDGE.data.feature_extractor import ImageFeatureExtractor
from experiments.SEDGE.models.backbone_factory import (
    create_all_backbones,
    DEFAULT_BACKBONES,
)


def evaluate(model, loader, device, name="Test"):
    model.eval()
    correct = 0
    total = 0
    all_weights = []
    all_outputs = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Evaluating {name}")
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, weights = model(inputs, return_weights=True)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_weights.append(weights.cpu())
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    acc = 100.0 * correct / total
    print(f"{name} Accuracy: {acc:.2f}%")

    # Analyze routing behavior
    avg_weights = torch.cat(all_weights, dim=0).mean(dim=0)
    print(f"{name} Avg Routing Weights: {avg_weights.tolist()}")

    # Calculate Weight Entropy (measure of router certainty)
    # entropy = -sum(p * log(p))
    avg_entropy = -torch.sum(avg_weights * torch.log(avg_weights + 1e-8)).item()
    print(f"{name} Router Weight Entropy: {avg_entropy:.4f}")

    # Calculate ECE
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    # Convert logits to probabilities
    probs = F.softmax(all_outputs, dim=1)
    ece = calculate_ece(probs, all_targets)
    print(f"{name} ECE: {ece:.4f}")

    return acc


def calculate_ece(probs, labels, n_bins=10):
    """Calculates Expected Calibration Error."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate SEDGE model")
    parser.add_argument("--sedge-checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="ImageNetSubset")
    args = parser.parse_args()

    device = get_device()
    num_classes = 10

    # 1. Load checkpoint metadata first
    print(f"\nLoading SEDGE checkpoint from {args.sedge_checkpoint}...")
    checkpoint = torch.load(
        args.sedge_checkpoint, map_location=device, weights_only=False
    )

    # Get architecture info from checkpoint
    backbone_names = checkpoint.get("backbones", DEFAULT_BACKBONES)
    num_classes = checkpoint.get("num_classes", 10)
    router_hidden_dims = checkpoint.get("router_hidden_dims", [64, 32])
    top_k = checkpoint.get("top_k", 0)

    print(f"  Backbones: {backbone_names}")
    print(f"  Router dims: {router_hidden_dims}, top_k: {top_k}")

    # 2. Load frozen pretrained backbones
    print("\nLoading frozen pretrained backbones...")
    backbone_models, _ = create_all_backbones(backbone_names, num_classes, device)

    # 3. Reconstruct SEDGE model with matching architecture
    feature_extractor = ImageFeatureExtractor()
    sedge_model = SEDGEModel(
        backbone_models,
        num_classes,
        feature_extractor,
        router_hidden_dims=router_hidden_dims,
        top_k=top_k,
    ).to(device)

    # 4. Load trained weights
    sedge_model.load_state_dict(checkpoint["model_state_dict"])
    print("  Checkpoint loaded successfully!")

    # 5. Evaluate across suites
    print("\n--- Evaluating on Clean Data ---")
    _, val_loader, _ = create_data_loaders(args.data_dir)
    evaluate(sedge_model, val_loader, device, "Clean Val")

    # Note: Synthetic and Phone-Photo suites would require specific directory layouts
    # or specific test loaders.
    try:
        print("\n--- Evaluating on Phone-Photo Data ---")
        phone_loader = create_test_loader(
            args.data_dir
        )  # Assuming 'test' folder is phones
        evaluate(sedge_model, phone_loader, device, "Phone-Photo")
    except Exception as e:
        print(f"Skipping Phone-Photo evaluation: {e}")


if __name__ == "__main__":
    main()
