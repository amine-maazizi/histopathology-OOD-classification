"""Training helpers extracted from the notebook."""

from __future__ import annotations

from contextlib import nullcontext
from itertools import combinations
import time

import torch
from tqdm.notebook import tqdm


def binary_accuracy(pred, target):
    """Compute the notebook's binary accuracy metric."""

    pred = (pred > 0.5).int()
    target = target.int()
    if target.ndim < pred.ndim:
        target = target.view_as(pred)
    return (pred == target).float().mean()


def _batch_correct_count(pred, target):
    pred_labels = pred.detach() > 0.5
    target_labels = target.detach().to(dtype=torch.bool)
    if target_labels.ndim < pred_labels.ndim:
        target_labels = target_labels.view_as(pred_labels)
    return pred_labels.eq(target_labels).sum()


def _clone_state_dict_cpu(state_dict):
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _clone_named_parameters_cpu(named_parameters):
    return {name: parameter.detach().cpu().clone() for name, parameter in named_parameters}


def _autocast_context(device, enabled):
    if not enabled or device.type != "cuda":
        return nullcontext()

    try:
        return torch.amp.autocast(device_type="cuda")
    except AttributeError:  # pragma: no cover - older torch fallback
        return torch.cuda.amp.autocast()


def _grad_scaler(device, enabled):
    if not enabled or device.type != "cuda":
        return None

    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # pragma: no cover - older torch fallback
        return torch.cuda.amp.GradScaler()


def _log_batch_timing(prefix, batch_index, transfer_time, forward_time, backward_time):
    print(
        f"{prefix} batch {batch_index + 1}: "
        f"data {transfer_time * 1000:.1f}ms | "
        f"forward {forward_time * 1000:.1f}ms | "
        f"backward/step {backward_time * 1000:.1f}ms"
    )


def _epoch_stats(total_loss, total_correct, total_examples):
    total_examples = max(int(total_examples), 1)
    return {
        "loss": total_loss / total_examples,
        "metric": total_correct / total_examples,
        "examples": total_examples,
    }


def train_epoch(
    train_dataloader,
    linear_probing,
    criterion,
    optimizer,
    device,
    frozen=None,
    use_amp=False,
    scaler=None,
    log_batch_timing=False,
):
    """Run one training epoch and return mean loss / metric statistics."""

    linear_probing.train()
    if frozen is not None:
        frozen.eval()

    scaler = scaler or _grad_scaler(device, use_amp)
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    for batch_index, (train_x, train_y) in enumerate(tqdm(train_dataloader, leave=False)):
        optimizer.zero_grad(set_to_none=True)

        if log_batch_timing:
            transfer_start = time.perf_counter()
        train_x = train_x.to(device, non_blocking=True)
        train_y = train_y.to(device, non_blocking=True)
        if log_batch_timing:
            transfer_time = time.perf_counter() - transfer_start

        if log_batch_timing:
            forward_start = time.perf_counter()
        with _autocast_context(device, use_amp):
            train_pred = linear_probing(train_x)
            train_target = train_y.float().view_as(train_pred)
        with _autocast_context(device, False):
            loss = criterion(train_pred.float(), train_target.float())
        if log_batch_timing:
            forward_time = time.perf_counter() - forward_start

        if log_batch_timing:
            backward_start = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if log_batch_timing:
            backward_time = time.perf_counter() - backward_start

        batch_size = int(train_y.shape[0])
        total_examples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_correct += float(_batch_correct_count(train_pred, train_y).item())

        if log_batch_timing and batch_index < 3:
            _log_batch_timing("Train", batch_index, transfer_time, forward_time, backward_time)

    return _epoch_stats(total_loss, total_correct, total_examples)


@torch.inference_mode()
def validate_epoch(val_dataloader, linear_probing, criterion, device, use_amp=False):
    """Run one validation epoch and return mean loss / metric statistics."""

    linear_probing.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    for val_x, val_y in tqdm(val_dataloader, leave=False):
        val_x = val_x.to(device, non_blocking=True)
        val_y = val_y.to(device, non_blocking=True)

        with _autocast_context(device, use_amp):
            val_pred = linear_probing(val_x)
            val_target = val_y.float().view_as(val_pred)
        with _autocast_context(device, False):
            loss = criterion(val_pred.float(), val_target.float())

        batch_size = int(val_y.shape[0])
        total_examples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_correct += float(_batch_correct_count(val_pred, val_y).item())

    return _epoch_stats(total_loss, total_correct, total_examples)


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
    use_amp=False,
    log_batch_timing=False,
):
    """Train with early stopping and save the best weights to disk."""

    best_val_metric, best_epoch = -float("inf"), 0
    best_state_dict = _clone_state_dict_cpu(linear_probing.state_dict())
    history = []
    scaler = _grad_scaler(device, use_amp)

    for epoch in range(num_epochs):
        train_stats = train_epoch(
            train_dataloader,
            linear_probing,
            criterion,
            optimizer,
            device,
            frozen=frozen,
            use_amp=use_amp,
            scaler=scaler,
            log_batch_timing=log_batch_timing,
        )
        print(
            f"Epoch train [{epoch + 1}/{num_epochs}] | Loss {train_stats['loss']:.4f} | Metric {train_stats['metric']:.4f}"
        )

        val_stats = validate_epoch(val_dataloader, linear_probing, criterion, device, use_amp=use_amp)
        print(
            f"Epoch valid [{epoch + 1}/{num_epochs}] | Loss {val_stats['loss']:.4f} | Metric {val_stats['metric']:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "train_metric": train_stats["metric"],
                "val_loss": val_stats["loss"],
                "val_metric": val_stats["metric"],
            }
        )

        if best_val_metric < val_stats["metric"]:
            print(f"New best accuracy {best_val_metric:.4f} -> {val_stats['metric']:.4f}")
            best_val_metric = val_stats["metric"]
            best_epoch = epoch
            best_state_dict = _clone_state_dict_cpu(linear_probing.state_dict())
            torch.save(best_state_dict, checkpoint_path)

        if epoch - best_epoch == patience:
            break

    linear_probing.load_state_dict(best_state_dict)
    return history


def train_epoch_frozen_backbone(
    train_dataloader,
    feature_extractor,
    linear_probing,
    criterion,
    optimizer,
    device,
    use_amp=False,
    scaler=None,
    log_batch_timing=False,
):
    """Run one epoch with a frozen feature extractor and trainable linear probe."""

    feature_extractor.eval()
    linear_probing.train()
    scaler = scaler or _grad_scaler(device, use_amp)
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    for batch_index, (train_x, train_y) in enumerate(tqdm(train_dataloader, leave=False)):
        optimizer.zero_grad(set_to_none=True)

        if log_batch_timing:
            transfer_start = time.perf_counter()
        train_x = train_x.to(device, non_blocking=True)
        train_y = train_y.to(device, non_blocking=True)
        if log_batch_timing:
            transfer_time = time.perf_counter() - transfer_start

        if log_batch_timing:
            forward_start = time.perf_counter()
        with torch.no_grad():
            with _autocast_context(device, use_amp):
                train_features = feature_extractor(train_x)
        with _autocast_context(device, use_amp):
            train_pred = linear_probing(train_features)
            train_target = train_y.float().view_as(train_pred)
        with _autocast_context(device, False):
            loss = criterion(train_pred.float(), train_target.float())
        if log_batch_timing:
            forward_time = time.perf_counter() - forward_start

        if log_batch_timing:
            backward_start = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if log_batch_timing:
            backward_time = time.perf_counter() - backward_start

        batch_size = int(train_y.shape[0])
        total_examples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_correct += float(_batch_correct_count(train_pred, train_y).item())

        if log_batch_timing and batch_index < 3:
            _log_batch_timing("Frozen train", batch_index, transfer_time, forward_time, backward_time)

    return _epoch_stats(total_loss, total_correct, total_examples)


@torch.inference_mode()
def validate_epoch_frozen_backbone(val_dataloader, feature_extractor, linear_probing, criterion, device, use_amp=False):
    """Validate with frozen backbone and linear probe."""

    feature_extractor.eval()
    linear_probing.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    for val_x, val_y in tqdm(val_dataloader, leave=False):
        val_x = val_x.to(device, non_blocking=True)
        val_y = val_y.to(device, non_blocking=True)

        with _autocast_context(device, use_amp):
            val_features = feature_extractor(val_x)
            val_pred = linear_probing(val_features)
            val_target = val_y.float().view_as(val_pred)
        with _autocast_context(device, False):
            loss = criterion(val_pred.float(), val_target.float())

        batch_size = int(val_y.shape[0])
        total_examples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_correct += float(_batch_correct_count(val_pred, val_y).item())

    return _epoch_stats(total_loss, total_correct, total_examples)


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
    use_amp=False,
    log_batch_timing=False,
):
    """Train a linear probe while keeping the backbone frozen and out of autograd."""

    best_val_metric, best_epoch = -float("inf"), 0
    best_state_dict = _clone_state_dict_cpu(linear_probing.state_dict())
    history = []
    scaler = _grad_scaler(device, use_amp)

    for epoch in range(num_epochs):
        train_stats = train_epoch_frozen_backbone(
            train_dataloader,
            feature_extractor,
            linear_probing,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
            scaler=scaler,
            log_batch_timing=log_batch_timing,
        )
        print(
            f"Epoch train [{epoch + 1}/{num_epochs}] | Loss {train_stats['loss']:.4f} | Metric {train_stats['metric']:.4f}"
        )

        val_stats = validate_epoch_frozen_backbone(val_dataloader, feature_extractor, linear_probing, criterion, device, use_amp=use_amp)
        print(
            f"Epoch valid [{epoch + 1}/{num_epochs}] | Loss {val_stats['loss']:.4f} | Metric {val_stats['metric']:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "train_metric": train_stats["metric"],
                "val_loss": val_stats["loss"],
                "val_metric": val_stats["metric"],
            }
        )

        if best_val_metric < val_stats["metric"]:
            print(f"New best accuracy {best_val_metric:.4f} -> {val_stats['metric']:.4f}")
            best_val_metric = val_stats["metric"]
            best_epoch = epoch
            best_state_dict = _clone_state_dict_cpu(linear_probing.state_dict())
            torch.save(best_state_dict, checkpoint_path)

        if epoch - best_epoch == patience:
            break

    linear_probing.load_state_dict(best_state_dict)
    return history


def train_epoch_trainable_backbone(
    train_dataloader,
    feature_extractor,
    linear_probing,
    criterion,
    optimizer,
    device,
    use_amp=False,
    scaler=None,
    log_batch_timing=False,
):
    """Run one epoch with trainable backbone LoRA adapters and linear probe."""

    feature_extractor.train()
    linear_probing.train()
    scaler = scaler or _grad_scaler(device, use_amp)
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    for batch_index, (train_x, train_y) in enumerate(tqdm(train_dataloader, leave=False)):
        optimizer.zero_grad(set_to_none=True)

        if log_batch_timing:
            transfer_start = time.perf_counter()
        train_x = train_x.to(device, non_blocking=True)
        train_y = train_y.to(device, non_blocking=True)
        if log_batch_timing:
            transfer_time = time.perf_counter() - transfer_start

        if log_batch_timing:
            forward_start = time.perf_counter()
        with _autocast_context(device, use_amp):
            train_features = feature_extractor(train_x)
            train_pred = linear_probing(train_features)
            train_target = train_y.float().view_as(train_pred)
        with _autocast_context(device, False):
            loss = criterion(train_pred.float(), train_target.float())
        if log_batch_timing:
            forward_time = time.perf_counter() - forward_start

        if log_batch_timing:
            backward_start = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if log_batch_timing:
            backward_time = time.perf_counter() - backward_start

        batch_size = int(train_y.shape[0])
        total_examples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_correct += float(_batch_correct_count(train_pred, train_y).item())

        if log_batch_timing and batch_index < 3:
            _log_batch_timing("Trainable", batch_index, transfer_time, forward_time, backward_time)

    return _epoch_stats(total_loss, total_correct, total_examples)


@torch.inference_mode()
def validate_epoch_trainable_backbone(val_dataloader, feature_extractor, linear_probing, criterion, device, use_amp=False):
    """Validate with trainable LoRA backbone and linear probe."""

    feature_extractor.eval()
    linear_probing.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    for val_x, val_y in tqdm(val_dataloader, leave=False):
        val_x = val_x.to(device, non_blocking=True)
        val_y = val_y.to(device, non_blocking=True)

        with _autocast_context(device, use_amp):
            val_features = feature_extractor(val_x)
            val_pred = linear_probing(val_features)
            val_target = val_y.float().view_as(val_pred)
        with _autocast_context(device, False):
            loss = criterion(val_pred.float(), val_target.float())

        batch_size = int(val_y.shape[0])
        total_examples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_correct += float(_batch_correct_count(val_pred, val_y).item())

    return _epoch_stats(total_loss, total_correct, total_examples)


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
    use_amp=False,
    log_batch_timing=False,
):
    """Fine-tune LoRA adapters and train linear probe."""

    best_val_loss, best_epoch = float("inf"), 0
    best_lora_state = _clone_named_parameters_cpu((name, parameter) for name, parameter in feature_extractor.named_parameters() if "lora_" in name)
    best_head_state = _clone_state_dict_cpu(linear_probing.state_dict())
    history = []
    scaler = _grad_scaler(device, use_amp)

    for epoch in range(num_epochs):
        train_stats = train_epoch_trainable_backbone(
            train_dataloader,
            feature_extractor,
            linear_probing,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
            scaler=scaler,
            log_batch_timing=log_batch_timing,
        )
        print(
            f"Epoch train [{epoch + 1}/{num_epochs}] | Loss {train_stats['loss']:.4f} | Metric {train_stats['metric']:.4f}"
        )

        val_stats = validate_epoch_trainable_backbone(val_dataloader, feature_extractor, linear_probing, criterion, device, use_amp=use_amp)
        print(
            f"Epoch valid [{epoch + 1}/{num_epochs}] | Loss {val_stats['loss']:.4f} | Metric {val_stats['metric']:.4f}"
        )

        if scheduler is not None:
            scheduler.step()
            print(f"LR: {scheduler.get_last_lr()}")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "train_metric": train_stats["metric"],
                "val_loss": val_stats["loss"],
                "val_metric": val_stats["metric"],
            }
        )

        if val_stats["loss"] < best_val_loss:
            print(f"New best loss {best_val_loss:.4f} -> {val_stats['loss']:.4f}")
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            best_lora_state = _clone_named_parameters_cpu((name, parameter) for name, parameter in feature_extractor.named_parameters() if "lora_" in name)
            best_head_state = _clone_state_dict_cpu(linear_probing.state_dict())
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
    use_amp=False,
    scaler=None,
    log_batch_timing=False,
):
    """Run one epoch with LoRA training and class-conditional CORAL regularization."""

    feature_extractor.train()
    linear_probing.train()
    scaler = scaler or _grad_scaler(device, use_amp)
    total_loss = 0.0
    total_correct = 0.0
    total_bce_loss = 0.0
    total_coral_loss = 0.0
    total_examples = 0

    for batch_index, (train_x, train_y, train_centers) in enumerate(tqdm(train_dataloader, leave=False)):
        optimizer.zero_grad(set_to_none=True)

        if log_batch_timing:
            transfer_start = time.perf_counter()
        train_x = train_x.to(device, non_blocking=True)
        train_y = train_y.to(device, non_blocking=True)
        train_centers = train_centers.to(device, non_blocking=True)
        if log_batch_timing:
            transfer_time = time.perf_counter() - transfer_start

        if log_batch_timing:
            forward_start = time.perf_counter()
        with _autocast_context(device, use_amp):
            train_features = feature_extractor(train_x)
            train_pred = linear_probing(train_features)
            train_target = train_y.float().view_as(train_pred)
        with _autocast_context(device, False):
            bce_loss = criterion(train_pred.float(), train_target.float())
            coral_loss = class_conditional_coral_loss(train_features.float(), train_y, train_centers, eps=coral_eps)
            loss = bce_loss + coral_lambda * coral_loss
        if log_batch_timing:
            forward_time = time.perf_counter() - forward_start

        if log_batch_timing:
            backward_start = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if log_batch_timing:
            backward_time = time.perf_counter() - backward_start

        batch_size = int(train_y.shape[0])
        total_examples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_bce_loss += bce_loss.detach().item() * batch_size
        total_coral_loss += coral_loss.detach().item() * batch_size
        total_correct += float(_batch_correct_count(train_pred, train_y).item())

        if log_batch_timing and batch_index < 3:
            _log_batch_timing("CORAL train", batch_index, transfer_time, forward_time, backward_time)

    stats = _epoch_stats(total_loss, total_correct, total_examples)
    stats["bce_loss"] = total_bce_loss / max(int(total_examples), 1)
    stats["coral_loss"] = total_coral_loss / max(int(total_examples), 1)
    return stats


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
    use_amp=False,
    log_batch_timing=False,
):
    """Fine-tune LoRA adapters with class-conditional CORAL and BCE validation early stopping."""

    best_val_loss, best_epoch = float("inf"), 0
    best_lora_state = _clone_named_parameters_cpu((name, parameter) for name, parameter in feature_extractor.named_parameters() if "lora_" in name)
    best_head_state = _clone_state_dict_cpu(linear_probing.state_dict())
    history = []
    scaler = _grad_scaler(device, use_amp)

    for epoch in range(num_epochs):
        train_stats = train_epoch_trainable_backbone_with_coral(
            train_dataloader,
            feature_extractor,
            linear_probing,
            criterion,
            optimizer,
            device,
            coral_lambda=coral_lambda,
            coral_eps=coral_eps,
            use_amp=use_amp,
            scaler=scaler,
            log_batch_timing=log_batch_timing,
        )
        print(
            f"Epoch train [{epoch + 1}/{num_epochs}] | Loss {train_stats['loss']:.4f} | Metric {train_stats['metric']:.4f}"
        )

        val_stats = validate_epoch_trainable_backbone(val_dataloader, feature_extractor, linear_probing, criterion, device, use_amp=use_amp)
        print(
            f"Epoch valid [{epoch + 1}/{num_epochs}] | Loss {val_stats['loss']:.4f} | Metric {val_stats['metric']:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "train_bce_loss": train_stats["bce_loss"],
                "train_coral_loss": train_stats["coral_loss"],
                "train_metric": train_stats["metric"],
                "val_loss": val_stats["loss"],
                "val_metric": val_stats["metric"],
            }
        )

        if val_stats["loss"] < best_val_loss:
            print(f"New best loss {best_val_loss:.4f} -> {val_stats['loss']:.4f}")
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            best_lora_state = _clone_named_parameters_cpu((name, parameter) for name, parameter in feature_extractor.named_parameters() if "lora_" in name)
            best_head_state = _clone_state_dict_cpu(linear_probing.state_dict())
            torch.save({"lora": best_lora_state, "head": best_head_state}, checkpoint_path)

        if epoch - best_epoch == patience:
            break

    feature_extractor.load_state_dict(best_lora_state, strict=False)
    linear_probing.load_state_dict(best_head_state)
    return history