"""Training helpers extracted from the notebook."""

from __future__ import annotations

import copy
from itertools import combinations

import torch
from tqdm.notebook import tqdm


def binary_accuracy(pred, target):
    """Compute the notebook's binary accuracy metric."""

    pred = (pred > 0.5).int()
    target = target.int()
    if target.ndim < pred.ndim:
        target = target.view_as(pred)
    return (pred == target).float().mean()


def train_epoch(train_dataloader, linear_probing, criterion, optimizer, device, frozen=None):
    """Run one training epoch and return per-sample losses and accuracies."""

    linear_probing.train()
    if frozen is not None:
        frozen.eval()
    train_metrics, train_losses = [], []

    for train_x, train_y in tqdm(train_dataloader, leave=False):
        optimizer.zero_grad()
        train_pred = linear_probing(train_x.to(device))
        train_target = train_y.to(device).float().view_as(train_pred)
        loss = criterion(train_pred, train_target)
        loss.backward()
        optimizer.step()

        train_losses.extend([loss.item()] * len(train_y))
        train_metric = binary_accuracy(train_pred.detach().cpu(), train_y.int().cpu())
        train_metrics.extend([train_metric.item()] * len(train_y))

    return train_metrics, train_losses


@torch.no_grad()
def validate_epoch(val_dataloader, linear_probing, criterion, device):
    """Run one validation epoch and return per-sample losses and accuracies."""

    linear_probing.eval()
    val_metrics, val_losses = [], []

    for val_x, val_y in tqdm(val_dataloader, leave=False):
        val_pred = linear_probing(val_x.to(device))
        val_target = val_y.to(device).float().view_as(val_pred)
        loss = criterion(val_pred, val_target)
        val_losses.extend([loss.item()] * len(val_y))
        val_metric = binary_accuracy(val_pred.detach().cpu(), val_y.int().cpu())
        val_metrics.extend([val_metric.item()] * len(val_y))

    return val_metrics, val_losses


def fit(
    train_dataloader,
    val_dataloader,
    linear_probing,
    optimizer,
    criterion,
    device,
    num_epochs=100,
    patience=10,
    checkpoint_path="best_model.pth",
    frozen=None,
):
    """Train with early stopping and save the best weights to disk."""

    min_loss, best_epoch = float("inf"), 0
    best_state_dict = copy.deepcopy(linear_probing.state_dict())
    history = []

    for epoch in range(num_epochs):
        train_metrics, train_losses = train_epoch(train_dataloader, linear_probing, criterion, optimizer, device, frozen=frozen)
        print(
            f"Epoch train [{epoch + 1}/{num_epochs}] | Loss {sum(train_losses) / len(train_losses):.4f} | Metric {sum(train_metrics) / len(train_metrics):.4f}"
        )

        val_metrics, val_losses = validate_epoch(val_dataloader, linear_probing, criterion, device)
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

        if mean_val_loss < min_loss:
            print(f"New best loss {min_loss:.4f} -> {mean_val_loss:.4f}")
            min_loss = mean_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(linear_probing.state_dict())
            torch.save(linear_probing.state_dict(), checkpoint_path)

        if epoch - best_epoch == patience:
            break

    linear_probing.load_state_dict(best_state_dict)
    return history


def train_epoch_frozen_backbone(train_dataloader, feature_extractor, linear_probing, criterion, optimizer, device):
    """Run one epoch with a frozen feature extractor and trainable linear probe."""

    feature_extractor.eval()
    linear_probing.train()
    train_metrics, train_losses = [], []

    for train_x, train_y in tqdm(train_dataloader, leave=False):
        optimizer.zero_grad()
        train_x = train_x.to(device, non_blocking=True)
        with torch.no_grad():
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


@torch.no_grad()
def validate_epoch_frozen_backbone(val_dataloader, feature_extractor, linear_probing, criterion, device):
    """Validate with frozen backbone and linear probe."""

    feature_extractor.eval()
    linear_probing.eval()
    val_metrics, val_losses = [], []

    for val_x, val_y in tqdm(val_dataloader, leave=False):
        val_x = val_x.to(device, non_blocking=True)
        val_features = feature_extractor(val_x)
        val_pred = linear_probing(val_features)
        val_target = val_y.to(device, non_blocking=True).float().view_as(val_pred)
        loss = criterion(val_pred, val_target)

        val_losses.extend([loss.item()] * len(val_y))
        val_metric = binary_accuracy(val_pred.detach().cpu(), val_y.int().cpu())
        val_metrics.extend([val_metric.item()] * len(val_y))

    return val_metrics, val_losses


def fit_frozen_backbone(
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
):
    """Train a linear probe while keeping the backbone frozen and out of autograd."""

    min_loss, best_epoch = float("inf"), 0
    best_state_dict = copy.deepcopy(linear_probing.state_dict())
    history = []

    for epoch in range(num_epochs):
        train_metrics, train_losses = train_epoch_frozen_backbone(
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

        val_metrics, val_losses = validate_epoch_frozen_backbone(
            val_dataloader,
            feature_extractor,
            linear_probing,
            criterion,
            device,
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

        if mean_val_loss < min_loss:
            print(f"New best loss {min_loss:.4f} -> {mean_val_loss:.4f}")
            min_loss = mean_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(linear_probing.state_dict())
            torch.save(linear_probing.state_dict(), checkpoint_path)

        if epoch - best_epoch == patience:
            break

    linear_probing.load_state_dict(best_state_dict)
    return history


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


@torch.no_grad()
def validate_epoch_trainable_backbone(val_dataloader, feature_extractor, linear_probing, criterion, device):
    """Validate with trainable LoRA backbone and linear probe."""

    feature_extractor.eval()
    linear_probing.eval()
    val_metrics, val_losses = [], []

    for val_x, val_y in tqdm(val_dataloader, leave=False):
        val_x = val_x.to(device, non_blocking=True)
        val_features = feature_extractor(val_x)
        val_pred = linear_probing(val_features)
        val_target = val_y.to(device, non_blocking=True).float().view_as(val_pred)
        loss = criterion(val_pred, val_target)

        val_losses.extend([loss.item()] * len(val_y))
        val_metric = binary_accuracy(val_pred.detach().cpu(), val_y.int().cpu())
        val_metrics.extend([val_metric.item()] * len(val_y))

    return val_metrics, val_losses

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
):
    """Fine-tune LoRA adapters and train linear probe."""

    min_loss, best_epoch = float("inf"), 0
    best_lora_state = copy.deepcopy({n: p for n, p in feature_extractor.named_parameters() if "lora_" in n})
    best_head_state = copy.deepcopy(linear_probing.state_dict())
    history = []

    for epoch in range(num_epochs):
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

        val_metrics, val_losses = validate_epoch_trainable_backbone(
            val_dataloader,
            feature_extractor,
            linear_probing,
            criterion,
            device,
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

        if mean_val_loss < min_loss:
            print(f"New best loss {min_loss:.4f} -> {mean_val_loss:.4f}")
            min_loss = mean_val_loss
            best_epoch = epoch
            best_lora_state = copy.deepcopy({n: p for n, p in feature_extractor.named_parameters() if "lora_" in n})
            best_head_state = copy.deepcopy(linear_probing.state_dict())
            torch.save({"lora": best_lora_state, "head": best_head_state}, checkpoint_path)

        if epoch - best_epoch == patience:
            break

    feature_extractor.load_state_dict(best_lora_state, strict=False)
    linear_probing.load_state_dict(best_head_state)
    return history


def covariance_matrix(features, eps=1e-5):
    """Compute a stabilized covariance matrix for a batch of features."""

    centered = features - features.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / (features.shape[0] - 1)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return cov + eps * eye


def class_conditional_coral_loss(features, labels, centers, eps=1e-5):
    """Compute CORAL over centers independently for each class."""

    labels = labels.view(-1).long()
    centers = centers.view(-1).long()
    loss = features.new_tensor(0.0)

    for class_id in (0, 1):
        class_covariances = []
        for center_id in torch.unique(centers):
            group_mask = (labels == class_id) & (centers == center_id)
            group_features = features[group_mask]
            if group_features.shape[0] >= 2:
                class_covariances.append(covariance_matrix(group_features, eps=eps))

        if len(class_covariances) < 2:
            continue

        for cov_a, cov_b in combinations(class_covariances, 2):
            loss = loss + torch.sum((cov_a - cov_b) ** 2)

    return loss


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


def fit_trainable_backbone_with_class_conditional_coral(
    train_dataloader,
    val_dataloader,
    feature_extractor,
    linear_probing,
    optimizer,
    criterion,
    device,
    coral_lambda=0.05,
    coral_eps=1e-5,
    num_epochs=100,
    patience=10,
    checkpoint_path="best_model.pth",
):
    """Fine-tune LoRA adapters with class-conditional CORAL and BCE validation early stopping."""

    min_loss, best_epoch = float("inf"), 0
    best_lora_state = copy.deepcopy({n: p for n, p in feature_extractor.named_parameters() if "lora_" in n})
    best_head_state = copy.deepcopy(linear_probing.state_dict())
    history = []

    for epoch in range(num_epochs):
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
        print(
            f"Epoch train [{epoch + 1}/{num_epochs}] | Loss {sum(train_losses) / len(train_losses):.4f} | Metric {sum(train_metrics) / len(train_metrics):.4f}"
        )

        val_metrics, val_losses = validate_epoch_trainable_backbone(
            val_dataloader,
            feature_extractor,
            linear_probing,
            criterion,
            device,
        )
        mean_val_loss = sum(val_losses) / len(val_losses)
        print(
            f"Epoch valid [{epoch + 1}/{num_epochs}] | Loss {mean_val_loss:.4f} | Metric {sum(val_metrics) / len(val_metrics):.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": sum(train_losses) / len(train_losses),
                "train_bce_loss": sum(train_bce_losses) / len(train_bce_losses),
                "train_coral_loss": sum(train_coral_losses) / len(train_coral_losses),
                "train_metric": sum(train_metrics) / len(train_metrics),
                "val_loss": mean_val_loss,
                "val_metric": sum(val_metrics) / len(val_metrics),
            }
        )

        if mean_val_loss < min_loss:
            print(f"New best loss {min_loss:.4f} -> {mean_val_loss:.4f}")
            min_loss = mean_val_loss
            best_epoch = epoch
            best_lora_state = copy.deepcopy({n: p for n, p in feature_extractor.named_parameters() if "lora_" in n})
            best_head_state = copy.deepcopy(linear_probing.state_dict())
            torch.save({"lora": best_lora_state, "head": best_head_state}, checkpoint_path)

        if epoch - best_epoch == patience:
            break

    feature_extractor.load_state_dict(best_lora_state, strict=False)
    linear_probing.load_state_dict(best_head_state)
    return history