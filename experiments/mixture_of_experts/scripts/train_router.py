#!/usr/bin/env python3
"""
Training script for the Mixture-of-Experts Router.
Freezes the experts and trains a dynamic routing network on the validation subset.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms


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

from experiments.base_ensemble.src.utils.device import (
    get_device,
    set_seed,
    get_device_name,
)
from experiments.base_ensemble.src.models import load_checkpoint
from experiments.base_ensemble.src.data import get_val_transform
from experiments.mixture_of_experts.src.dataset import get_router_datasets
from experiments.mixture_of_experts.src.model import (
    MoERouter,
    FeatureWrapper,
    MoEEnsemble,
)


def get_hard_router_transform():
    """
    Heavy augmentation pipeline to simulate phone-captured artifacts.
    This helps the router see where experts fail during training.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))], p=0.4
            ),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
            ),
            transforms.RandomApply(
                [
                    transforms.RandomChoice(
                        [
                            transforms.RandomRotation(degrees=30),
                            transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
                        ]
                    )
                ],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train MoE Router")
    parser.add_argument(
        "--data-dir", "--data_dir", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        type=str,
        default="experiments/base_ensemble/checkpoints",
        help="Where experts are stored",
    )
    parser.add_argument(
        "--save-dir",
        "--save_dir",
        type=str,
        default="experiments/mixture_of_experts/checkpoints",
        help="Where to save router",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    # Experts to use (paths relative to checkpoint-dir or absolute)
    parser.add_argument(
        "--experts",
        nargs="+",
        default=[
            "best_densenet121.pth",
            "best_resnet34.pth",
            "best_efficientnet_b0.pth",
            "best_vit_b_16.pth",
        ],
    )

    # Feature provider for router
    parser.add_argument(
        "--fe-idx",
        type=int,
        default=3,
        help="Index of expert to provide features (default 3: ViT)",
    )
    parser.add_argument(
        "--fe-dim",
        type=int,
        default=768,
        help="Dimension of router input features (ViT: 768, ResNet34: 512, DenseNet121: 1024, EffNet: 1280)",
    )

    # Temperature Calibration
    parser.add_argument(
        "--temp-file",
        "--temp_file",
        type=str,
        default=None,
        help="Path to temperatures.pth from calibration",
    )

    return parser.parse_args()


def train_router():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print("\n" + "=" * 60)
    print(" Mixture-of-Experts Router Training")
    print(f" Device: {get_device_name(device)}")
    print("=" * 60)

    # 1. Load experts and temperatures
    print(f"\n Loading {len(args.experts)} experts...")
    experts = []
    checkpoint_dir = Path(args.checkpoint_dir)

    # Load temperatures if provided
    temp_dict = {}
    if args.temp_file and os.path.exists(args.temp_file):
        print(f" Loading temperatures from {args.temp_file}")
        temp_dict = torch.load(args.temp_file, map_location=device)
    elif args.temp_file:
        print(
            f" Warning: Temperature file {args.temp_file} not found. Using T=1.0 for all."
        )

    expert_temps = []
    for exp_path in args.experts:
        path = (
            checkpoint_dir / exp_path if not os.path.isabs(exp_path) else Path(exp_path)
        )
        model, name, _ = load_checkpoint(str(path), device)

        # Wrapped for feature extraction
        wrapped = FeatureWrapper(model, name)
        wrapped.eval()
        for param in wrapped.parameters():
            param.requires_grad = False
        experts.append(wrapped)

        # Match temperature
        t = temp_dict.get(name, 1.0)
        expert_temps.append(t)
        print(f"   ✓ Loaded {name} (T={t:.3f}) from {path.name}")

    # 2. Setup Router and Ensemble
    router = MoERouter(input_dim=args.fe_dim, num_experts=len(experts)).to(device)
    moe_model = MoEEnsemble(
        experts, router, feature_provider_idx=args.fe_idx, temperatures=expert_temps
    ).to(device)

    # 3. Setup Data (80/20 split of val set with hard training aug)
    train_transform = get_hard_router_transform()
    val_transform = get_val_transform()
    train_ds, val_ds, class_names = get_router_datasets(
        args.data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        seed=args.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"\n Data Split:")
    print(f"   Router Train: {len(train_ds)} samples")
    print(f"   Router Val:   {len(val_ds)} samples")

    # 4. Training loop
    optimizer = optim.Adam(router.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        router.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits, weights = moe_model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                loss=train_loss / (pbar.n + 1), acc=100.0 * correct / total
            )

        # Validation
        router.eval()
        val_loss = 0.0
        v_correct = 0
        v_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, weights = moe_model(inputs)
                loss = criterion(logits, targets)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                v_total += targets.size(0)
                v_correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * v_correct / v_total
        print(
            f"   Epoch {epoch + 1} Summary: Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "router_state_dict": router.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "config": vars(args),
                },
                save_path / "best_router.pth",
            )
            print(f"   --> Saved new best router (acc: {val_acc:.2f}%)")

    print("\n" + "=" * 60)
    print(f" Training Complete! Best Router Val Acc: {best_val_acc:.2f}%")
    print(f" Checkpoint saved to: {save_path}/best_router.pth")
    print("=" * 60)


if __name__ == "__main__":
    train_router()
