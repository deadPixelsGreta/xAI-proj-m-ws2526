"""Dataset utilities for ImageNetSubset-style folder layouts."""

import os
from typing import Tuple, Optional

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Class names for ImageNetSubset (alphabetical order as loaded by ImageFolder)
CLASS_NAMES = [
    "binder",
    "coffee_mug",
    "computer_keyboard",
    "mouse",
    "notebook",
    "remote_control",
    "soup_bowl",
    "teapot",
    "toilet_tissue",
    "wooden_spoon",
]

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_resnet18_augmentation(aug_params: dict) -> list:
    """ColorJitter + geometric augmentation for ResNet18."""
    brightness = aug_params.get('brightness', 0.3)
    contrast = aug_params.get('contrast', 0.3)
    saturation = aug_params.get('saturation', 0.3)
    hue = aug_params.get('hue', 0.1)
    rotation = aug_params.get('rotation', 15)
    
    return [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ),
        transforms.RandomRotation(degrees=rotation),
        transforms.RandomGrayscale(p=0.1),
    ]


def get_resnet34_augmentation(aug_params: dict) -> list:
    """AutoAugment/TrivialAugment for ResNet34."""
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
        base.append(transforms.AutoAugment(policy=policy_map.get(policy, transforms.AutoAugmentPolicy.IMAGENET)))
    
    return base


def get_resnet50_augmentation(aug_params: dict) -> list:
    """RandAugment for ResNet50."""
    num_ops = aug_params.get('num_ops', 2)
    magnitude = aug_params.get('magnitude', 9)
    use_cutout = aug_params.get('use_cutout', False)
    
    augs = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
    ]
    
    if use_cutout:
        augs.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)))
    
    return augs


# Augementation transforms for training depending on model type
def get_train_transform(model_name: str, aug_params: Optional[dict] = None) -> transforms.Compose:
    """Return training transforms with model-specific augmentation.

    Args:
        model_name: Name of the model for model-specific augmentation
        aug_params: Optional dict with augmentation parameters for sweep
    Returns:
        Composed training transforms
    """
    # Use sweep parameters if provided, otherwise use defaults
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
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_val_transform() -> transforms.Compose:
    """Return validation/inference transforms without augmentation."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def create_data_loaders(
    data_dir: str, 
    batch_size: int = 32, 
    num_workers: int = 4, 
    pin_memory: bool = True, 
    model_name: str = "resnet18",
    aug_params: Optional[dict] = None
) -> Tuple[DataLoader, DataLoader, int]:
    """Create data loaders from an ImageFolder layout under data_dir/train and data_dir/val.

    Args:
        data_dir: Root directory containing train/val subdirectories
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory in DataLoader
        model_name: Name of the model for model-specific augmentation
        aug_params: Optional dict with augmentation parameters for sweep

    Returns (train_loader, val_loader, num_classes).
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transform(model_name, aug_params))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, len(train_dataset.classes)


def get_dataset_info(data_dir: str) -> dict:
    """Return basic dataset stats: counts, class list, and class count."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=None)
    val_dataset = datasets.ImageFolder(val_dir, transform=None)

    return {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "classes": train_dataset.classes,
        "num_classes": len(train_dataset.classes),
    }


def get_test_transform() -> transforms.Compose:
    """Return test transforms (identical to validation)."""
    return get_val_transform()


def create_test_loader(
    data_dir: str, batch_size: int = 32, num_workers: int = 4, pin_memory: bool = True
) -> DataLoader:
    """Create data loader from an ImageFolder layout under data_dir/test.

    Returns test_loader.
    """
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    test_dataset = datasets.ImageFolder(test_dir, transform=get_test_transform())

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader
