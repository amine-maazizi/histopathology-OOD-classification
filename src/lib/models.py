"""Model helpers extracted from the notebook."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

def load_dinov2_backbone():
    """Load the DINOv2 backbone used by the notebook baseline."""

    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


def build_linear_probing_head(feature_extractor):
    """Build the linear probing head used in the notebook."""

    return nn.Sequential(
        nn.Linear(feature_extractor.num_features, 1),
        nn.Sigmoid(),
    )


def build_lora_dinov2(rank=8, alpha=1.0):
    """Build DINOv2 for LoRA fine-tuning. Loads the DINOv2 backbone and injects LoRA adapters into each attention block's qkv and projection layers."""

    backbone = load_dinov2_backbone()
    for param in backbone.parameters():
        param.requires_grad_(False)

    for block in backbone.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha)
        block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha)
        
    return backbone

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with LoRA."""

    def __init__(self, linear, rank=8, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank

        for param in self.linear.parameters():
            param.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)

