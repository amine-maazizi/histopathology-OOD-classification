"""Training helpers extracted from the notebook and further implemented."""

from __future__ import annotations

import copy

import torch
from tqdm.notebook import tqdm

from .losses import class_conditional_coral_loss


def binary_accuracy(pred, target):
    """Compute the notebook's binary accuracy metric."""

    pred = (pred > 0.5).int()
    target = target.int()
    if target.ndim < pred.ndim:
        target = target.view_as(pred)
    return (pred == target).float().mean()


def train_epoch_frozen_backbone(train_dataloader, linear_probing, criterion, optimizer, device, feature_extractor=None):
    """
    Run one training epoch for linear probing with a frozen backbone.
    Used for all solutions with frozen backbones. If a `feature_extractor` is provided,
    features are computed on-the-fly, which is used for solutions with augmentations,
    since augmentations differ each batch and features cannot be precomputed.
    Otherwise, the dataloader is expected to pass precomputed features directly.

    """

    if feature_extractor is not None:
        feature_extractor.eval()
    linear_probing.train()
    train_metrics, train_losses = [], []

    for train_x, train_y in tqdm(train_dataloader, leave=False):
        optimizer.zero_grad()
        train_x = train_x.to(device, non_blocking=True)

        # When using augmentations, they differ each batch so features
        # cannot be precomputed and must be computed on-the-fly.
        if feature_extractor is not None:
            with torch.no_grad():
                train_x = feature_extractor(train_x)
        train_pred = linear_probing(train_x)
        train_target = train_y.to(device, non_blocking=True).float().view_as(train_pred)
        
        loss = criterion(train_pred, train_target)
        loss.backward()
        optimizer.step()

        train_losses.extend([loss.item()] * len(train_y))
        train_metric = binary_accuracy(train_pred.detach().cpu(), train_y.int().cpu())
        train_metrics.extend([train_metric.item()] * len(train_y))

    return train_metrics, train_losses

def train_epoch_trainable_backbone(train_dataloader, feature_extractor, linear_probing, criterion, optimizer, device):
    """Run one epoch with trainable backbone LoRA adapters and linear probe."""

    feature_extractor.train()
    linear_probing.train()
    train_metrics, train_losses = [], []

    for train_x, train_y in tqdm(train_dataloader, leave=False):
        optimizer.zero_grad()
        train_x = train_x.to(device, non_blocking=True)
        train_features = feature_extractor(train_x)
        train_pred = linear_probing(train_features)
        train_target = train_y.to(device, non_blocking=True).float().view_as(train_pred)
        loss = criterion(train_pred, train_target)
        loss.backward()
        optimizer.step()

        train_losses.extend([loss.item()] * len(train_y))
        train_metric = binary_accuracy(train_pred.detach().cpu(), train_y.int().cpu())
        train_metrics.extend([train_metric.item()] * len(train_y))

    return train_metrics, train_losses

def train_epoch_trainable_backbone_with_coral(
    train_dataloader,
    feature_extractor,
    linear_probing,
    criterion,
    optimizer,
    device,
    coral_lambda=0.05,
    coral_eps=1e-5,
):
    """Run one epoch with LoRA training and class-conditional CORAL regularization."""

    feature_extractor.train()
    linear_probing.train()
    train_metrics, train_losses, train_bce_losses, train_coral_losses = [], [], [], []

    for train_x, train_y, train_centers in tqdm(train_dataloader, leave=False):
        optimizer.zero_grad()

        train_x = train_x.to(device, non_blocking=True)
        train_y = train_y.to(device, non_blocking=True)
        train_centers = train_centers.to(device, non_blocking=True)

        train_features = feature_extractor(train_x)
        train_pred = linear_probing(train_features)
        train_target = train_y.float().view_as(train_pred)

        # Compute class-conditional CORAL loss and sum with BCE loss for total
        bce_loss = criterion(train_pred, train_target)
        coral_loss = class_conditional_coral_loss(train_features, train_y, train_centers, eps=coral_eps)
        loss = bce_loss + coral_lambda * coral_loss

        loss.backward()
        optimizer.step()

        train_losses.extend([loss.item()] * len(train_y))
        train_bce_losses.extend([bce_loss.item()] * len(train_y))
        train_coral_losses.extend([coral_loss.item()] * len(train_y))
        train_metric = binary_accuracy(train_pred.detach().cpu(), train_y.int().cpu())
        train_metrics.extend([train_metric.item()] * len(train_y))

    return train_metrics, train_losses, train_bce_losses, train_coral_losses


@torch.no_grad()
def validate(val_dataloader, linear_probing, criterion, device, feature_extractor=None):
    """
    Run one validation epoch for all solutions. If a `feature_extractor` is provided,
    features are computed on-the-fly, which is used for trainable backbone and 
    frozen backbone with augmentations solutions. Otherwise, the dataloader is expected
    to pass precomputed features directly.

    """

    if feature_extractor is not None:
        feature_extractor.eval()
    linear_probing.eval()
    val_metrics, val_losses = [], []

    for val_x, val_y in tqdm(val_dataloader, leave=False):
        val_x = val_x.to(device, non_blocking=True)
        if feature_extractor is not None:
            val_x = feature_extractor(val_x)
        val_pred = linear_probing(val_x)
        val_target = val_y.to(device, non_blocking=True).float().view_as(val_pred)
        loss = criterion(val_pred, val_target)

        val_losses.extend([loss.item()] * len(val_y))
        val_metric = binary_accuracy(val_pred.detach().cpu(), val_y.int().cpu())
        val_metrics.extend([val_metric.item()] * len(val_y))

    return val_metrics, val_losses


def fit_frozen_backbone(
    train_dataloader,
    val_dataloader,
    linear_probing,
    optimizer,
    criterion,
    device,
    num_epochs=100,
    patience=10,
    checkpoint_path="best_model.pth",
    feature_extractor=None,
):
    """
    Train a linear classification head on top of a frozen backbone. Used for all solutions
    with frozen backbones. Performs full training procedure, validation, early stopping
    based on validation accuracy, checkpointing and loading of the best model. If a 
    `feature_extractor` is provided, it is kept frozen and used only for feature extraction.

    """

    min_loss, best_epoch, max_val_acc = float("inf"), 0, -float("inf")
    best_state_dict = copy.deepcopy(linear_probing.state_dict())
    history = []

    for epoch in range(num_epochs):
        train_metrics, train_losses = train_epoch_frozen_backbone(
            train_dataloader,
            linear_probing,
            criterion,
            optimizer,
            device,
            feature_extractor=feature_extractor,
        )
        print(
            f"Epoch train [{epoch + 1}/{num_epochs}] | Loss {sum(train_losses) / len(train_losses):.4f} | Metric {sum(train_metrics) / len(train_metrics):.4f}"
        )

        val_metrics, val_losses = validate(
            val_dataloader,
            linear_probing,
            criterion,
            device,
            feature_extractor=feature_extractor,
        )
        mean_val_loss = sum(val_losses) / len(val_losses)
        print(
            f"Epoch valid [{epoch + 1}/{num_epochs}] | Loss {mean_val_loss:.4f} | Metric {sum(val_metrics) / len(val_metrics):.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": sum(train_losses) / len(train_losses),
                "train_metric": sum(train_metrics) / len(train_metrics),
                "val_loss": mean_val_loss,
                "val_metric": sum(val_metrics) / len(val_metrics),
            }
        )

        if max_val_acc < sum(val_metrics) / len(val_metrics):
            print(f"New best accuracy {max_val_acc:.4f}: {sum(val_metrics) / len(val_metrics):.4f}")
            max_val_acc = sum(val_metrics) / len(val_metrics)
            best_epoch = epoch
            best_state_dict = copy.deepcopy(linear_probing.state_dict())
            torch.save(linear_probing.state_dict(), checkpoint_path)

        # Early stopping based on validation accuracy
        if epoch - best_epoch == patience:
            break

    # Load best model before running test evaluation
    linear_probing.load_state_dict(best_state_dict)
    return history

def fit_trainable_backbone(
    train_dataloader,
    val_dataloader,
    feature_extractor,
    linear_probing,
    optimizer,
    criterion,
    device,
    num_epochs=100,
    patience=10,
    checkpoint_path="best_model.pth",
    scheduler=None,
    coral_lambda=None,
    coral_eps=1e-5,
):
    """
    Train end-to-end with LoRA adapters in the backbone and a linear classification head. 
    Used for all solutions with trainable backbones. Performs full training procedure, 
    validation, optimizer scheduling, early stopping based on validation loss, checkpointing
    and loading of the best model. If `coral_lambda` is provided, class-conditional CORAL
    regularization is applied with the given lambda weight and epsilon.

    """

    min_loss, best_epoch = float("inf"), 0
    best_lora_state = copy.deepcopy({n: p for n, p in feature_extractor.named_parameters() if "lora_" in n})
    best_head_state = copy.deepcopy(linear_probing.state_dict())
    history = []

    for epoch in range(num_epochs):
        if coral_lambda is not None:
            # Train with class-conditional CORAL regularization
            train_metrics, train_losses, train_bce_losses, train_coral_losses = train_epoch_trainable_backbone_with_coral(
                train_dataloader,
                feature_extractor,
                linear_probing,
                criterion,
                optimizer,
                device,
                coral_lambda=coral_lambda,
                coral_eps=coral_eps,
            )
        else:
            # Train base LoRA model (without CORAL)
            train_metrics, train_losses = train_epoch_trainable_backbone(
                train_dataloader,
                feature_extractor,
                linear_probing,
                criterion,
                optimizer,
                device,
            )
        print(
            f"Epoch train [{epoch + 1}/{num_epochs}] | Loss {sum(train_losses) / len(train_losses):.4f} | Metric {sum(train_metrics) / len(train_metrics):.4f}"
        )

        val_metrics, val_losses = validate(
            val_dataloader,
            linear_probing,
            criterion,
            device,
            feature_extractor=feature_extractor,
        )
        mean_val_loss = sum(val_losses) / len(val_losses)
        print(
            f"Epoch valid [{epoch + 1}/{num_epochs}] | Loss {mean_val_loss:.4f} | Metric {sum(val_metrics) / len(val_metrics):.4f}"
        )

        # Apply scheduler step
        if scheduler is not None:
            scheduler.step()

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": sum(train_losses) / len(train_losses),
            "train_metric": sum(train_metrics) / len(train_metrics),
            "val_loss": mean_val_loss,
            "val_metric": sum(val_metrics) / len(val_metrics),
        }

        # Add CORAL-specific metrics to history
        if coral_lambda is not None:
            history_entry["train_bce_loss"] = sum(train_bce_losses) / len(train_bce_losses)
            history_entry["train_coral_loss"] = sum(train_coral_losses) / len(train_coral_losses)
        history.append(history_entry)

        if mean_val_loss < min_loss:
            print(f"New best loss {min_loss:.4f} -> {mean_val_loss:.4f}")
            min_loss = mean_val_loss
            best_epoch = epoch
            best_lora_state = copy.deepcopy({n: p for n, p in feature_extractor.named_parameters() if "lora_" in n})
            best_head_state = copy.deepcopy(linear_probing.state_dict())
            torch.save({"backbone": best_lora_state, "head": best_head_state}, checkpoint_path)

        # Early stopping based on validation loss
        if epoch - best_epoch == patience:
            break

    # Load best model before running test evaluation
    feature_extractor.load_state_dict(best_lora_state, strict=False)
    linear_probing.load_state_dict(best_head_state)
    return history


