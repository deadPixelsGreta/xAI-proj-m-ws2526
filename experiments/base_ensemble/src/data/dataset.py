"""Dataset utilities for ImageNetSubset-style folder layouts."""

import os
from typing import Tuple

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


def get_train_transform(sota: bool = True) -> transforms.Compose:
    """Return training transforms with resize, crop, flip, and ImageNet normalization.

    Args:
        sota: If True, uses TrivialAugmentWide (SOTA). If False, uses legacy RandAugment.
    """
    if sota:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
    sota_aug: bool = True,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create data loaders from an ImageFolder layout under data_dir/train and data_dir/val.

    Returns (train_loader, val_loader, num_classes).
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(
        train_dir, transform=get_train_transform(sota=sota_aug)
    )
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
