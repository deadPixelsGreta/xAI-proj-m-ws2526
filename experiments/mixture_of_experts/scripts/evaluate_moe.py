#!/usr/bin/env python3
"""
Evaluation script for the Mixture-of-Experts Ensemble.
Verifies performance on the target (test) set compared to static ensembling.
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
from experiments.base_ensemble.src.models import load_checkpoint
from experiments.base_ensemble.src.data import get_val_transform, CLASS_NAMES
from experiments.mixture_of_experts.src.model import MoERouter, FeatureWrapper, MoEEnsemble

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MoE Ensemble")
    parser.add_argument("--data-dir", "--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", type=str, default="experiments/base_ensemble/checkpoints", help="Where experts are stored")
    parser.add_argument("--router-path", type=str, default="experiments/mixture_of_experts/checkpoints/best_router.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    
    # Experts to use (paths relative to checkpoint-dir or absolute)
    parser.add_argument("--experts", nargs="+", default=[
        "best_densenet121_blur.pth",
        "best_resnet34_geometric.pth",
        "best_efficientnet_b0_color.pth",
        "best_vit_b_16_vit.pth"
    ])
    
    # Feature provider for router
    parser.add_argument("--fe-idx", type=int, default=3)
    parser.add_argument("--fe-dim", type=int, default=768)
    parser.add_argument("--temperatures-path", type=str, default="experiments/temp_calibration/checkpoints/temperatures.pth", help="Optional calibrated temperatures")

    return parser.parse_args()

def load_moe_ensemble_helper(expert_paths, checkpoint_dir, router_path, fe_idx, fe_dim, device, temperatures_path=None):
    """
    Helper function to build and load a fully calibrated MoE ensemble.
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # 1. Load experts
    experts = []
    for exp_path in expert_paths:
        path = checkpoint_dir / exp_path if not os.path.isabs(exp_path) else Path(exp_path)
        model, name, _ = load_checkpoint(str(path), device)
        wrapped = FeatureWrapper(model, name)
        wrapped.eval()
        experts.append(wrapped)
        
    # 2. Load router
    router = MoERouter(input_dim=fe_dim, num_experts=len(experts)).to(device)
    state = torch.load(router_path, map_location=device)
    router.load_state_dict(state['router_state_dict'])
    router.eval()
    
    # 3. Load temperatures
    temperatures = None
    if temperatures_path and os.path.exists(temperatures_path):
        temp_dict = torch.load(temperatures_path, map_location=device)
        temperatures = []
        for exp_path in expert_paths:
            path = checkpoint_dir / exp_path if not os.path.isabs(exp_path) else Path(exp_path)
            # Re-running load_checkpoint momentarily just for name (sub-optimal but safe)
            _, name, _ = load_checkpoint(str(path), "cpu")
            t = temp_dict.get(name, 1.0)
            temperatures.append(t)
            
    moe_model = MoEEnsemble(experts, router, feature_provider_idx=fe_idx, temperatures=temperatures).to(device)
    moe_model.eval()
    return moe_model

def evaluate_moe():
    args = parse_args()
    device = get_device()
    
    print("\n" + "="*60)
    print(f" Mixture-of-Experts Evaluation on {args.split.upper()} set")
    print(f" Device: {get_device_name(device)}")
    print("="*60)
    
    # 1. Load experts
    # 2. Load router
    # 3. Load temperatures if available
    
    moe_model = load_moe_ensemble_helper(
        expert_paths=args.experts,
        checkpoint_dir=args.checkpoint_dir,
        router_path=args.router_path,
        fe_idx=args.fe_idx,
        fe_dim=args.fe_dim,
        device=device,
        temperatures_path=args.temperatures_path
    )
    
    # 4. Setup Data
    transform = get_val_transform()
    test_dir = Path(args.data_dir) / args.split
    test_ds = datasets.ImageFolder(root=str(test_dir), transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # 5. Run Inference
    correct_moe = 0
    correct_mean = 0
    total = 0
    
    all_weights = []
    
    print(f"\n Running inference on {len(test_ds)} images...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # MoE prediction
            logits_moe, weights = moe_model(inputs)
            _, pred_moe = logits_moe.max(1)
            correct_moe += pred_moe.eq(targets).sum().item()
            
            all_weights.append(weights.cpu())
            
            # Simple mean comparison (baseline)
            expert_logits = [e(inputs)[0] for e in experts]
            mean_logits = torch.stack(expert_logits, dim=1).mean(dim=1)
            _, pred_mean = mean_logits.max(1)
            correct_mean += pred_mean.eq(targets).sum().item()
            
            total += targets.size(0)
            
    # 6. Results Summary
    moe_acc = 100. * correct_moe / total
    mean_acc = 100. * correct_mean / total
    
    avg_weights = torch.cat(all_weights, dim=0).mean(dim=0).tolist()
    expert_names = ["DenseNet121", "ResNet34", "EfficientNet", "ViT-B16"] # Match order in args.experts
    
    print("\n" + "-"*40)
    print(f" Performance Summary ({args.split.upper()}):")
    print("-"*40)
    print(f" Simple Mean Ensemble: {mean_acc:.2f}%")
    print(f" MoE Router Ensemble:  {moe_acc:6.2f}%")
    print(f" Improvement:          {moe_acc - mean_acc:+.2f}%")
    
    print("\n Average Routing Weights:")
    for name, w in zip(expert_names, avg_weights):
        print(f"   {name:15s}: {w:.4f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    evaluate_moe()
