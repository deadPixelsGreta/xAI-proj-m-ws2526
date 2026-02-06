import torch
from torchvision import transforms

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_base_transforms():
    """Foundational transforms required for all models."""
    return [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]

def get_blur_transforms():
    """Focus on Gaussian/Motion blur and JPEG compression artifacts."""
    return transforms.Compose(get_base_transforms() + [
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))], p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.2), # Mimics compression loss
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_color_transforms():
    """Focus on extreme Color Jittering, Solarization, and Contrast shifts."""
    return transforms.Compose(get_base_transforms() + [
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomSolarize(threshold=128, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_geometric_transforms():
    """Focus on Elastic transforms, Perspective shifts, and Random Cropping."""
    return transforms.Compose(get_base_transforms() + [
        transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.3), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_vit_transforms():
    """Standard strong augmentation for ViT (usually TrivialAugment or RandAugment)."""
    return transforms.Compose(get_base_transforms() + [
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transform():
    """Standard validation transforms."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

POLICY_MAP = {
    "blur": get_blur_transforms,
    "color": get_color_transforms,
    "geometric": get_geometric_transforms,
    "vit": get_vit_transforms,
}
