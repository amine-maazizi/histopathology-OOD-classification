import torch
import torchstain


class ReinhardNormalizer:
    """Wraps torchstain's ReinhardNormalizer as a preprocessing transform."""

    def __init__(self, reference):
        self.normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
        self.normalizer.fit(reference * 255.0)

    def __call__(self, x):
        norm = self.normalizer.normalize(I=x * 255.0)
        if norm.shape[-1] == 3:
            norm = norm.permute(2, 0, 1)
        return (norm.float() / 255.0).clamp(0.0, 1.0)


class StainColorJitter(object):
    """Data augmentation that randomly jitters the stain color of histopathology images."""
    
    def __init__(self, sigma=0.05):
        self.M = torch.tensor([
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
            [0.27, 0.57, 0.78]
            ]
        )
        self.M_inv = torch.inverse(self.M)
        self.eps = 1e-6
        self.sigma = sigma # strength of the jitter

    def __call__(self, x):
        _, H, W = x.shape
        work_x = x.to(dtype=torch.float32)

        M = self.M.to(device=work_x.device, dtype=work_x.dtype)
        M_inv = self.M_inv.to(device=work_x.device, dtype=work_x.dtype)
        eps = torch.tensor(self.eps, device=work_x.device, dtype=work_x.dtype)

        P_flat = work_x.permute(1, 2, 0).reshape(-1, 3)
        P_scaled = torch.clamp(255.0 * P_flat, min=eps)

        # TODO: Explain
        S = -(torch.log(P_scaled)) @ M_inv

        # alpha ~ U(1-sigma, 1+sigma), beta ~ U(-sigma, sigma)
        alpha = 1 + (torch.rand(3, device=work_x.device, dtype=work_x.dtype) - 0.5) * 2 * self.sigma
        beta = (torch.rand(3, device=work_x.device, dtype=work_x.dtype) - 0.5) * 2 * self.sigma

        # TODO: Explain
        S = S * alpha + beta

        # TODO: Explain
        P = torch.exp(-(S @ M)) - eps
        P = (P / 255.0).reshape(H, W, 3).permute(2, 0, 1)
        P = torch.clamp(P, 0.0, 1.0)

        return P
    

class RandomStainColorJitter(object):
    """Applies StainColorJitter augmentation randomly with a given probability."""

    def __init__(self, sigma=0.05, p=0.75):
        self.p = p
        self.jitter = StainColorJitter(sigma=sigma)

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            return self.jitter(x)
        return x
