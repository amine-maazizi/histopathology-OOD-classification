"Data augmentation and normalization transforms for histopathology images to improve performance across different staining conditions."

import torch
import torchstain


class ReinhardNormalizer:
    """
    Wraps torchstain's ReinhardNormalizer as a preprocessing transform.
    Converts images to LAB color space and matches mean and std of each channel to a reference image.
    
    """

    def __init__(self, reference):
        self.normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
        # All images are normalized with respect to the same reference image
        # so we fit once, and normalize them all later.
        self.normalizer.fit(reference * 255.0)

    def __call__(self, x):
        norm = self.normalizer.normalize(I=x * 255.0)
        if norm.shape[-1] == 3:
            norm = norm.permute(2, 0, 1)
        return (norm.float() / 255.0).clamp(0.0, 1.0)
    
    
class StainColorJitter(object):
    """
    Data augmentation that randomly jitters the stain color of histopathology images.
    Projects RGB to optical density (OD) space, applies random scaling and shifting to the OD values, and projects back to RGB space.
    Adapted from: https://github.com/i-gao/targeted-augs/blob/main/examples/data_augmentation/stain_color_jitter.py
    
    """
    
    def __init__(self, sigma=0.05):
        # Normalized OD matrix 
        self.M = torch.tensor([
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
            [0.27, 0.57, 0.78]
            ]
        )
        self.M_inv = torch.inverse(self.M)
        self.eps = 1e-6
        # Strength of the jitter
        self.sigma = sigma

    def __call__(self, x):
        _, H, W = x.shape
        x = x.to(dtype=torch.float32)

        M = self.M.to(device=x.device, dtype=x.dtype)
        M_inv = self.M_inv.to(device=x.device, dtype=x.dtype)
        eps = torch.tensor(self.eps, device=x.device, dtype=x.dtype)

        # Scale and flattening
        P_flat = x.permute(1, 2, 0).reshape(-1, 3)
        P_scaled = torch.clamp(255.0 * P_flat, min=eps)

        # Project from RGB to OD space
        S = -(torch.log(P_scaled)) @ M_inv

        # Sample scale alpha ~ U(1-sigma, 1+sigma) and shift beta ~ U(-sigma, sigma)
        alpha = 1 + (torch.rand(3, device=x.device, dtype=x.dtype) - 0.5) * 2 * self.sigma
        beta = (torch.rand(3, device=x.device, dtype=x.dtype) - 0.5) * 2 * self.sigma

        # Augment stain color in OD space
        S = S * alpha + beta

        # Convert back to RGB space
        P = torch.exp(-(S @ M)) - eps
        P = (P / 255.0).reshape(H, W, 3).permute(2, 0, 1)
        P = torch.clamp(P, 0.0, 1.0)

        return P
    
class RandomStainColorJitter(object):
    """Applies StainColorJitter augmentation randomly based on a given probability."""

    def __init__(self, sigma=0.05, p=0.75):
        self.p = p
        self.jitter = StainColorJitter(sigma=sigma)

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            return self.jitter(x)
        return x
