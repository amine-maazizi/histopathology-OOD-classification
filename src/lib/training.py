"""Training helpers extracted from the notebook."""

from __future__ import annotations

import copy

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
        loss = criterion(train_pred, train_y.to(device).float())
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
        loss = criterion(val_pred, val_y.to(device).float())
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
