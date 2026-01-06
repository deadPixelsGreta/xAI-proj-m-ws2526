"""Optimized dataset utilities for A100 GPU training with advanced augmentation."""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_resnet18_augmentation(aug_params: dict) -> list:
    """
    ColorJitter + geometric augmentation for ResNet18.
    Optimized for A100 with stronger augmentation.
    """
    brightness = aug_params.get('brightness', 0.3)
    contrast = aug_params.get('contrast', 0.3)
    saturation = aug_params.get('saturation', 0.3)
    hue = aug_params.get('hue', 0.1)
    rotation = aug_params.get('rotation', 15)
    
    return [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ),
        transforms.RandomRotation(degrees=rotation),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ]


def get_resnet34_augmentation(aug_params: dict) -> list:
    """
    AutoAugment/TrivialAugment for ResNet34.
    Policy-based augmentation optimized for medium models.
    """
    policy = aug_params.get('policy', 'IMAGENET')
    use_trivial = aug_params.get('use_trivial_augment', False)
    
    policy_map = {
        'IMAGENET': transforms.AutoAugmentPolicy.IMAGENET,
        'CIFAR10': transforms.AutoAugmentPolicy.CIFAR10,
        'SVHN': transforms.AutoAugmentPolicy.SVHN,
    }
    
    base = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    
    if use_trivial:
        base.append(transforms.TrivialAugmentWide())
    else:
        base.append(transforms.AutoAugment(
            policy=policy_map.get(policy, transforms.AutoAugmentPolicy.IMAGENET)
        ))
    
    return base


def get_resnet50_augmentation(aug_params: dict) -> list:
    """
    RandAugment + Random Erasing for ResNet50.
    Strong augmentation for larger models on A100.
    """
    num_ops = aug_params.get('num_ops', 2)
    magnitude = aug_params.get('magnitude', 9)
    use_cutout = aug_params.get('use_cutout', True)
    
    augs = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
    ]
    
    if use_cutout:
        augs.append(transforms.RandomErasing(
            p=0.25, 
            scale=(0.02, 0.33), 
            ratio=(0.3, 3.3)
        ))
    
    return augs


def get_train_transform(model_name: str, aug_params: Optional[dict] = None) -> transforms.Compose:
    """
    Create optimized training transforms for A100.
    
    Args:
        model_name: resnet18, resnet34, or resnet50
        aug_params: Augmentation parameters from sweep
    """
    if aug_params is None:
        aug_params = {}
    
    # Get model-specific augmentation
    if model_name == "resnet18":
        aug_transforms = get_resnet18_augmentation(aug_params)
    elif model_name == "resnet34":
        aug_transforms = get_resnet34_augmentation(aug_params)
    elif model_name == "resnet50":
        aug_transforms = get_resnet50_augmentation(aug_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return transforms.Compose(
        aug_transforms
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transform() -> transforms.Compose:
    """Validation transforms without augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_data_loaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    model_name: str