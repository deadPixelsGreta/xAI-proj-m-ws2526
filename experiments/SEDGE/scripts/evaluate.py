import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from experiments.bagging.src.models import load_checkpoint
from experiments.bagging.src.data import create_data_loaders, create_test_loader
from experiments.bagging.src.utils import get_device
from experiments.SEDGE.models.sedge import SEDGEModel
from experiments.SEDGE.data.feature_extractor import ImageFeatureExtractor


def evaluate(model, loader, device, name="Test"):
    model.eval()
    correct = 0
    total = 0
    all_weights = []

    pbar = tqdm(loader, desc=f"Evaluating {name}")
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, weights = model(inputs, return_weights=True)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_weights.append(weights.cpu())

    acc = 100.0 * correct / total
    print(f"{name} Accuracy: {acc:.2f}%")

    # Analyze routing behavior
    avg_weights = torch.cat(all_weights, dim=0).mean(dim=0)
    print(f"{name} Avg Routing Weights: {avg_weights.tolist()}")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate SEDGE model")
    parser.add_argument("--sedge-checkpoint", type=str, required=True)
    parser.add_argument("--backbones", type=str, nargs="+", required=True)
    parser.add_argument("--data-dir", type=str, default="ImageNetSubset")
    args = parser.parse_args()

    device = get_device()
    num_classes = 10

    # 1. Load backbones
    backbone_models = []
    for cp_path in args.backbones:
        model, _, _ = load_checkpoint(cp_path, device, num_classes=num_classes)
        backbone_models.append(model)

    # 2. Reconstruct SEDGE model
    feature_extractor = ImageFeatureExtractor()
    sedge_model = SEDGEModel(backbone_models, num_classes, feature_extractor).to(device)
    sedge_model.load_state_dict(torch.load(args.sedge_checkpoint, map_location=device))

    # 3. Evaluate across suites
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
