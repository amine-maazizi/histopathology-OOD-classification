"""Model helpers extracted from the notebook."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def load_dinov2_backbone():
    """Load the DINOv2 backbone used by the notebook baseline."""

    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


def load_uni_backbone(pretrained=True, **kwargs):
    """Load the UNI ViT-L/16 backbone via timm.

    Supports both online loading from Hugging Face and local checkpoint loading.
    When `checkpoint_path` is provided, the underlying timm model is created
    locally and the checkpoint is loaded directly into it.
    """

    model_name = kwargs.pop("model_name", "hf-hub:MahmoodLab/UNI")
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    img_size = kwargs.pop("img_size", 224)
    patch_size = kwargs.pop("patch_size", 16)
    init_values = kwargs.pop("init_values", 1e-5)
    dynamic_img_size = kwargs.pop("dynamic_img_size", True)

    try:
        import timm
    except ImportError as error:
        raise ImportError(
            "Loading UNI backbones requires the optional dependency 'timm'. "
            "Install it first, then authenticate to Hugging Face if loading "
            "MahmoodLab/UNI from hf-hub."
        ) from error

    if checkpoint_path is not None:
        architecture_name = model_name
        if architecture_name.startswith("hf-hub:"):
            architecture_name = "vit_large_patch16_224"

        backbone = timm.create_model(
            architecture_name,
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            init_values=init_values,
            num_classes=0,
            dynamic_img_size=dynamic_img_size,
            **kwargs,
        )
        backbone.load_state_dict(_torch_load(checkpoint_path), strict=True)
        setattr(backbone, "_resolved_backbone_name", architecture_name)
        setattr(backbone, "_requested_backbone_name", model_name)
        return backbone

    try:
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            init_values=init_values,
            dynamic_img_size=dynamic_img_size,
            num_classes=0,
            **kwargs,
        )
    except Exception as error:
        raise RuntimeError(
            "Failed to load the requested UNI backbone. "
            f"Requested model_name='{model_name}'. "
            "If you need offline loading, pass an explicit local checkpoint_path. "
            "UNI does not fall back to a generic ViT when model_name='uni'."
        ) from error

    setattr(backbone, "_resolved_backbone_name", model_name)
    setattr(backbone, "_requested_backbone_name", model_name)
    return backbone


def load_backbone(backbone_name="dinov2_vits14", pretrained=True, **kwargs):
    """Load a supported backbone by short name."""

    normalized_name = backbone_name.lower()
    if normalized_name in {"dinov2", "dinov2_vits14"}:
        return load_dinov2_backbone()
    if normalized_name == "uni":
        return load_uni_backbone(pretrained=pretrained, **kwargs)
    raise ValueError(f"Unsupported backbone_name '{backbone_name}'.")


def build_linear_probing_head(feature_extractor):
    """Build the linear probing head used in the notebook."""

    return nn.Sequential(
        nn.Linear(feature_extractor.num_features, 1),
        nn.Sigmoid(),
    )


def get_backbone_blocks(backbone):
    """Return the transformer blocks used for LoRA injection."""

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            f"Backbone {backbone.__class__.__name__} does not expose a 'blocks' attribute."
        )
    return blocks


def inject_lora_into_vit_attention(backbone, rank=8, alpha=1.0, lora_targets=("qkv", "proj")):
    """Replace attention qkv/proj layers with LoRA-wrapped linear layers."""

    allowed_targets = {"qkv", "proj"}
    lora_targets = tuple(lora_targets)
    if not lora_targets:
        raise ValueError("lora_targets must contain at least one attention sublayer name.")

    invalid_targets = sorted(set(lora_targets) - allowed_targets)
    if invalid_targets:
        raise ValueError(
            f"Unsupported lora_targets {invalid_targets}. Supported targets: {sorted(allowed_targets)}."
        )

    for block_index, block in enumerate(get_backbone_blocks(backbone)):
        attention = getattr(block, "attn", None)
        if attention is None:
            raise RuntimeError(
                f"Block {block_index} on {backbone.__class__.__name__} does not expose an attn module."
            )

        for attribute_name in lora_targets:
            linear = getattr(attention, attribute_name, None)
            if linear is None:
                raise RuntimeError(
                    f"Block {block_index} on {backbone.__class__.__name__} is missing attention.{attribute_name}."
                )
            if isinstance(linear, LoRALinear):
                continue
            setattr(attention, attribute_name, LoRALinear(linear, rank=rank, alpha=alpha))

    return backbone


def build_lora_backbone(backbone_name, rank=8, alpha=1.0, pretrained=True, lora_targets=("qkv", "proj"), **kwargs):
    """Load a backbone and inject LoRA adapters into its attention layers."""

    backbone = load_backbone(backbone_name, pretrained=pretrained, **kwargs)
    for param in backbone.parameters():
        param.requires_grad_(False)

    backbone = inject_lora_into_vit_attention(backbone, rank=rank, alpha=alpha, lora_targets=lora_targets)
    setattr(backbone, "_lora_targets", tuple(lora_targets))
    return backbone


def build_lora_dinov2(rank=8, alpha=1.0):
    """Backward-compatible helper that builds LoRA DINOv2."""

    return build_lora_backbone("dinov2_vits14", rank=rank, alpha=alpha)


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with LoRA."""

    def __init__(self, linear, rank=8, alpha=1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

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
