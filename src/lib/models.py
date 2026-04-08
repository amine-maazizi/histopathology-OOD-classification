"""Model helpers extracted from the notebook."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


BACKBONE_LOADERS = {}


def register_backbone(name):
    """Register a backbone loader under a normalized name."""

    def decorator(loader):
        BACKBONE_LOADERS[name.lower()] = loader
        return loader

    return decorator


def _unwrap_state_dict(checkpoint):
    """Extract a plain state_dict from common checkpoint formats."""

    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("state_dict", "model", "backbone"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            return state_dict

    return checkpoint


def _load_checkpoint_into_model(model, checkpoint_path, strict=True):
    """Load a checkpoint into a model while tolerating common wrapper formats."""

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")

    try:
        checkpoint = torch.load(checkpoint_file, weights_only=True, map_location="cpu")
    except TypeError:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")

    state_dict = _unwrap_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' did not match {model.__class__.__name__}. "
            f"Missing keys: {missing[:10]}; unexpected keys: {unexpected[:10]}."
        )


def _resolve_nested_attribute(module, attribute_path):
    current = module
    for attribute in attribute_path.split("."):
        current = getattr(current, attribute, None)
        if current is None:
            return None
    return current


def get_backbone_num_features(model):
    """Infer the feature dimension exposed by a backbone."""

    for attribute_name in ("num_features", "embed_dim", "hidden_size", "feature_dim"):
        value = getattr(model, attribute_name, None)
        if isinstance(value, int) and value > 0:
            return value

    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        return head.in_features

    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Linear):
        return classifier.in_features

    raise RuntimeError(
        f"Could not infer the output feature dimension for {model.__class__.__name__}. "
        "Add a small adapter in get_backbone_num_features() for this backbone."
    )


def get_backbone_blocks(model):
    """Return the transformer blocks used for LoRA injection."""

    for attribute_path in ("blocks", "encoder.blocks"):
        blocks = _resolve_nested_attribute(model, attribute_path)
        if blocks is not None:
            return blocks

    raise RuntimeError(
        f"Could not locate transformer blocks on {model.__class__.__name__}. "
        "LoRA injection currently expects a ViT-style model exposing model.blocks "
        "or model.encoder.blocks."
    )


@register_backbone("dinov2_vits14")
@register_backbone("dinov2")
def load_dinov2_backbone(pretrained=True, **kwargs):
    """Load the DINOv2 backbone used by the notebook baseline."""

    kwargs = dict(kwargs)
    kwargs.setdefault("pretrained", pretrained)
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", **kwargs)


@register_backbone("uni")
def load_uni_backbone(pretrained=True, **kwargs):
    """Load a UNI ViT-L/16 backbone.

    Supported modes:
    1. Hugging Face gated model through timm + hf-hub:
       model_name="hf-hub:MahmoodLab/UNI"
    2. Local checkpoint loading:
       checkpoint_path=".../pytorch_model.bin"
       with model_name defaulting to "vit_large_patch16_224"

    UNI requires LayerScale parameters, so init_values=1e-5 must be set when
    constructing the model.
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

    # Case 1: explicit local checkpoint path
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
        _load_checkpoint_into_model(backbone, checkpoint_path, strict=True)
        return backbone

    # Case 2: Hugging Face / timm model loading
    candidate_names = []
    if model_name:
        candidate_names.append(model_name)

    # Add fallbacks only if user did not already specify one of them
    for candidate in (
        "hf-hub:MahmoodLab/UNI",
        "hf-hub:MahmoodLab/uni",
        "vit_large_patch16_224",
    ):
        if candidate not in candidate_names:
            candidate_names.append(candidate)

    load_errors = []
    backbone = None

    for candidate_name in candidate_names:
        try:
            if candidate_name.startswith("hf-hub:"):
                backbone = timm.create_model(
                    candidate_name,
                    pretrained=pretrained,
                    init_values=init_values,
                    dynamic_img_size=dynamic_img_size,
                    **kwargs,
                )
            elif candidate_name == "vit_large_patch16_224":
                # Plain timm ViT-L/16 architecture. This is only useful if
                # pretrained=False or if the caller later loads a local checkpoint.
                backbone = timm.create_model(
                    candidate_name,
                    pretrained=pretrained,
                    img_size=img_size,
                    patch_size=patch_size,
                    init_values=init_values,
                    num_classes=0,
                    dynamic_img_size=dynamic_img_size,
                    **kwargs,
                )
            else:
                backbone = timm.create_model(
                    candidate_name,
                    pretrained=pretrained,
                    init_values=init_values,
                    dynamic_img_size=dynamic_img_size,
                    **kwargs,
                )
            break
        except Exception as error:  # pragma: no cover
            load_errors.append(f"{candidate_name}: {error}")

    if backbone is None:
        raise RuntimeError(
            "Could not load a UNI backbone. Tried: "
            + "; ".join(load_errors)
            + ". Use a Hugging Face-authorized account for "
            "'hf-hub:MahmoodLab/UNI', or pass "
            "config['backbone_kwargs']['checkpoint_path'] to a local "
            "UNI checkpoint, with init_values=1e-5."
        )

    return backbone


def load_backbone(backbone_name, pretrained=True, **kwargs):
    """Load a backbone from the registry."""

    try:
        backbone_loader = BACKBONE_LOADERS[backbone_name.lower()]
    except KeyError as error:
        available = ", ".join(sorted(BACKBONE_LOADERS))
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. Available backbones: {available}."
        ) from error

    return backbone_loader(pretrained=pretrained, **kwargs)


def build_linear_probing_head(feature_extractor, hidden_dim=None, dropout=0.0, use_sigmoid=True):
    """Build the linear or shallow-MLP probing head used in the notebook."""

    input_dim = get_backbone_num_features(feature_extractor)
    layers = []

    if hidden_dim is None:
        layers.append(nn.Linear(input_dim, 1))
    else:
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
            ]
        )

    if use_sigmoid:
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def inject_lora_into_vit_attention(backbone, rank=8, alpha=1.0):
    """Replace attention qkv/proj layers with LoRA-wrapped linear layers."""

    for block_index, block in enumerate(get_backbone_blocks(backbone)):
        attention = getattr(block, "attn", None)
        if attention is None:
            raise RuntimeError(
                f"Block {block_index} on {backbone.__class__.__name__} does not expose "
                "an attn module. LoRA injection currently expects a ViT-style attention "
                "block with qkv and proj layers."
            )

        for attribute_name in ("qkv", "proj"):
            linear = getattr(attention, attribute_name, None)
            if linear is None:
                raise RuntimeError(
                    f"Block {block_index} on {backbone.__class__.__name__} is missing "
                    f"attention.{attribute_name}. This backbone is not compatible with "
                    "the generic ViT LoRA injector yet."
                )
            if isinstance(linear, LoRALinear):
                continue
            setattr(attention, attribute_name, LoRALinear(linear, rank=rank, alpha=alpha))

    return backbone


def build_lora_backbone(backbone_name, rank=8, alpha=1.0, pretrained=True, **kwargs):
    """Load a backbone and inject LoRA adapters into its attention layers."""

    backbone = load_backbone(backbone_name, pretrained=pretrained, **kwargs)
    for param in backbone.parameters():
        param.requires_grad_(False)

    return inject_lora_into_vit_attention(backbone, rank=rank, alpha=alpha)


def build_lora_dinov2(rank=8, alpha=1.0, pretrained=True, **kwargs):
    """Backward-compatible DINOv2 helper for LoRA fine-tuning."""

    return build_lora_backbone(
        "dinov2_vits14",
        rank=rank,
        alpha=alpha,
        pretrained=pretrained,
        **kwargs,
    )


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a low-rank adapter."""

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