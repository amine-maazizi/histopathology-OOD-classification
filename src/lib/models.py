"""Model helpers extracted from the notebook."""

from __future__ import annotations

import torch
from torch import nn


def load_dinov2_backbone():
    """Load the DINOv2 backbone used by the notebook baseline."""

    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


def build_linear_probing_head(feature_extractor):
    """Build the linear probing head used in the notebook."""

    return nn.Sequential(
        nn.Linear(feature_extractor.num_features, 1),
        nn.Sigmoid(),
    )
