#!/usr/bin/env python3
"""
Evaluation script for the Mixture-of-Experts Ensemble with Test-Time Augmentation (TTA).
Final strategy to push performance on noisy phone-captured data.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

# Add project root to path
def setup_path():
    markers = {".git", "requirements.txt"}
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in markers):
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    return None

setup_path()

from experiments.base_ensemble.src.utils.device import get_device, get_device_name
from experiments.base_ensemble.src.data import get_val_transform
from experiments.mixture_of_experts.scripts.evaluate_moe import load_moe_ensemble_helper
from experiments.tta.src.wrapper import TTAWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MoE Ensemble with TTA")
    parser.add_argument("--data-dir", "--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", type=str, default="experiments/base_ensemble/checkpoints")
    parser.add_argument("--router-path", type=str, default="experiments/mixture_of_experts/checkpoints/best_router.pth")
    parser.add_argument("--temperatures-path", type=str, default="experiments/temp_calibration/checkpoints/temperatures.pth")
    parser.add_argument("--batch-size", type=int, default=16) # Lower batch size due to N samples in memory
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tta-samples", type=int, default=8, help="Number of TTA samples per image")
    
    parser.add_argument("--experts", nargs="+", default=[
        "best_densenet121_blur.pth",
        "best_resnet34_geometric.pth",
        "best_efficientnet_b0_color.pth",
        "best_vit_b_16_vit.pth"
    ])
    parser.add_argument("--fe-idx", type=int, default=3)
    parser.add_argument("--fe-dim", type=int, default=768)

    return parser.parse_args()

def evaluate_tta():
    args = parse_args()
    device = get_device()
    
    print("\n" + "="*60)
    print(f" MoE + TTA Evaluation on {args.split.upper()} set")
    print(f" Samples: {args.tta_samples} | Device: {get_device_name(device)}")
    print("="*60)
    
    # 1. Load MoE Ensemble
    base_moe = load_moe_ensemble_helper(
        expert_paths=args.experts,
        checkpoint_dir=args.checkpoint_dir,
        router_path=args.router_path,
        fe_idx=args.fe_idx,
        fe_dim=args.fe_dim,
        device=device,
        temperatures_path=args.temperatures_path
    )
    
    # 2. Wrap with TTA
    model = TTAWrapper(base_moe, num_samples=args.tta_samples).to(device)
    model.eval()
    
    # 3. Setup Data (Back to stable 2-sample TTA: Original + Flip)
    transform = get_val_transform()
    test_dir = Path(args.data_dir) / args.split
    test_ds = datasets.ImageFolder(root=str(test_dir), transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # 4. Inference Loop
    correct = 0
    total = 0
    
    print(f"\n Running Stable 2-Sample TTA (Original + Flip) on {len(test_ds)} images...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Use the TTAWrapper which is currently set to Original + Flip
            log_probs, _ = model(inputs)
            _, preds = log_probs.max(1)
            
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)



            
    acc = 100. * correct / total
    print("\n" + "-"*40)
    print(f" FINAL MoE + TTA Accuracy: {acc:.2f}%")
    print("-"*40)
    print("\n" + "="*60)

if __name__ == "__main__":
    evaluate_tta()
