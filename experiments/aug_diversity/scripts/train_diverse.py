#!/usr/bin/env python3
"""CLI to train models with diversity augmentation (Path A)."""

import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# Ensure 'experiments' folder is in the python path
def setup_path():
    """Find and add project root to sys.path."""
    markers = {".git", "requirements.txt"}
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in markers):
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    return None

setup_path()

from experiments.base_ensemble.src.utils.device import get_device, set_seed, get_device_name
from experiments.base_ensemble.src.utils.arguments_parsing import add_common_args, add_wandb_args, add_model_args
from experiments.base_ensemble.src.models import SUPPORTED_MODELS, create_model
from experiments.base_ensemble.src.training import train, TrainingConfig
from experiments.aug_diversity.src.augmentations import POLICY_MAP, get_val_transform

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Ensemble Members with Unique Augmentation Policies (Diversity Training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Common arguments (data-dir, checkpoint-dir)
    add_common_args(parser)
    add_model_args(parser, SUPPORTED_MODELS)
    add_wandb_args(parser, default_project="aug-diversity")
    
    # Diversity-specific arguments
    parser.add_argument(
        "--policy", 
        type=str, 
        choices=list(POLICY_MAP.keys()), 
        required=True, 
        help="The augmentation policy to specialize this model with (e.g., blur, color, geometric, vit)"
    )
    
    # Training overrides
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32, dest="batch_size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    
    print("\n" + "="*60)
    print(f" Diversity Training: Path A")
    print(f" Model:  {args.model}")
    print(f" Policy: {args.policy.upper()}")
    print(f" Device: {get_device_name(device)}")
    print("="*60)
    
    # 1. Setup specialized data transforms
    train_transform = POLICY_MAP[args.policy]()
    val_transform = get_val_transform()
    
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        sys.exit(1)
        
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    num_classes = len(train_dataset.classes)
    
    # 2. Create model and move to device
    model = create_model(args.model, num_classes=num_classes, pretrained=not args.no_pretrained)
    model = model.to(device)
    
    # 3. Setup training config using from_dict for robustness
    config_dict = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_classes": num_classes,
        "seed": args.seed,
        "use_sota_aug": False, # We handle our own transforms here
        "amp": args.amp,
        "grad_clip": args.grad_clip,
        "accum_steps": args.accum_steps,
        "scheduler": args.scheduler
    }
    config = TrainingConfig.from_dict(config_dict)
    
    # 4. Run training using the existing trainer logic
    save_dir = args.checkpoint_dir or "checkpoints"
    
    wandb_config = {
        "project": args.wandb_project,
        "run_name": f"{args.model}-{args.policy}-seed{args.seed}",
        "group": "aug-diversity",
        "job_type": "train",
        "notes": f"Specialized model with {args.policy} augmentation policy"
    }
    
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=save_dir,
        model_name=f"{args.model}_{args.policy}",
        wandb_enabled=args.wandb,
        wandb_config=wandb_config
    )
    
    print("\nTraining Complete!")

if __name__ == "__main__":
    main()
