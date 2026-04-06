import torch

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
        C, H, W = x.shape
        P_flat = x.permute(1, 2, 0).reshape(-1, 3)

        # TODO: Explain
        S = -(torch.log(255.0 * P_flat + self.eps)) @ self.M_inv

        # alpha ~ U(1-sigma, 1+sigma), beta ~ U(-sigma, sigma)
        alpha = 1 + (torch.rand(3, device=x.device, dtype=x.dtype) - 0.5) * 2 * self.sigma
        beta = (torch.rand(3, device=x.device, dtype=x.dtype) - 0.5) * 2 * self.sigma

        # TODO: Explain
        S = S * alpha + beta

        # TODO: Explain
        P = torch.exp(-(S @ self.M)) - self.eps
        P = (P / 255.0).reshape(H, W, 3).permute(2, 0, 1)
        P = torch.clamp(P, 0.0, 1.0)

        return P