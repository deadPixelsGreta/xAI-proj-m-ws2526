import torch
import numpy as np
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import PIL.ImageFilter as ImageFilter
from torchvision import transforms
import io

class JPEGCompression:
    def __init__(self, quality_range=(10, 50)):
        self.quality_range = quality_range
    
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        
        quality = np.random.randint(self.quality_range[0], self.quality_range[1] + 1)
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality)
        output.seek(0)
        return Image.open(output)

class MotionBlur:
    def __init__(self, kernel_size=(3, 9)):
        self.kernel_size = kernel_size
    
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        
        size = np.random.randint(self.kernel_size[0], self.kernel_size[1] + 1)
        # Simple approximation of motion blur using a 1D kernel
        return img.filter(ImageFilter.GaussianBlur(radius=size/3)) # Simplified for PIL

class SensorNoise:
    def __init__(self, std_range=(0.01, 0.1)):
        self.std_range = std_range
    
    def __call__(self, img):
        # Noise is easier on tensors
        if isinstance(img, Image.Image):
            img_tensor = transforms.ToTensor()(img)
        else:
            img_tensor = img
            
        std = np.random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(img_tensor) * std
        img_noisy = torch.clamp(img_tensor + noise, 0, 1)
        
        return transforms.ToPILImage()(img_noisy)

def get_broadened_corruptions():
    """Returns a list of corruption transforms that simulate phone capture."""
    return [
        transforms.RandomApply([JPEGCompression()], p=0.5),
        transforms.RandomApply([MotionBlur()], p=0.3),
        transforms.RandomApply([SensorNoise()], p=0.4),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    ]

def get_robust_transform(base_transform):
    """Wraps a base transform with broadened corruptions."""
    return transforms.Compose(get_broadened_corruptions() + [base_transform])
