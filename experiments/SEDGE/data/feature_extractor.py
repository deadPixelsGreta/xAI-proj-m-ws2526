import torch
import torch.nn.functional as F
import numpy as np


class ImageFeatureExtractor:
    """Extracts domain-agnostic image degradation features."""

    @staticmethod
    def get_blur_feature(img_tensor):
        """Computes Laplacian variance as a blur indicator."""
        # Convert to grayscale if RGB
        if img_tensor.shape[1] == 3:
            gray = (
                0.299 * img_tensor[:, 0, :, :]
                + 0.587 * img_tensor[:, 1, :, :]
                + 0.114 * img_tensor[:, 2, :, :]
            )
            gray = gray.unsqueeze(1)
        else:
            gray = img_tensor

        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32,
            device=img_tensor.device,
        ).expand(1, 1, 3, 3)

        laplacian_map = F.conv2d(gray, laplacian_kernel, padding=1)
        variance = torch.var(laplacian_map, dim=(1, 2, 3))
        return variance.unsqueeze(1)  # [B, 1]

    @staticmethod
    def get_noise_feature(img_tensor):
        """Estimates noise level using local variance."""
        avg_kernel = torch.ones((1, 1, 3, 3), device=img_tensor.device) / 9.0

        # Grayscale
        if img_tensor.shape[1] == 3:
            gray = (
                0.299 * img_tensor[:, 0, :, :]
                + 0.587 * img_tensor[:, 1, :, :]
                + 0.114 * img_tensor[:, 2, :, :]
            )
            gray = gray.unsqueeze(1)
        else:
            gray = img_tensor

        mu = F.conv2d(gray, avg_kernel, padding=1)
        mu2 = F.conv2d(gray**2, avg_kernel, padding=1)
        local_var = torch.relu(mu2 - mu**2)

        noise_level = torch.mean(local_var, dim=(1, 2, 3))
        return noise_level.unsqueeze(1)  # [B, 1]

    @staticmethod
    def get_blockiness_feature(img_tensor):
        """Rough proxy for JPEG blockiness by looking at 8x8 grid gradients."""
        # Simplified: look at horizontal and vertical gradients at 8-pixel intervals
        diff_h = torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1])
        diff_v = torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :])

        # Check if gradients at multiples of 8 are significantly different
        # For simplicity, we just return the mean gradient magnitude
        # Truly estimating blockiness is complex, this is a proxy.
        grad_mag = torch.mean(diff_h, dim=(1, 2, 3)) + torch.mean(diff_v, dim=(1, 2, 3))
        return grad_mag.unsqueeze(1)  # [B, 1]

    def __call__(self, img_tensor):
        """Extracts all features and concatenates them."""
        blur = self.get_blur_feature(img_tensor)
        noise = self.get_noise_feature(img_tensor)
        blocks = self.get_blockiness_feature(img_tensor)

        # Normalize features? Usually good for MLPs
        # For now, just concatenate
        features = torch.cat([blur, noise, blocks], dim=1)
        return features
