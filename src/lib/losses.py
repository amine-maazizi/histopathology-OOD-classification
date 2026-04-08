"""Custom loss functions implemented for additional training."""

from __future__ import annotations
from itertools import combinations
import torch

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