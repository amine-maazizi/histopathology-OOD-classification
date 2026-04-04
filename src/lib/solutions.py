"""Solution registry and baseline orchestration."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .datasets import BaselineDataset, PrecomputedDataset, precompute
from .inference import load_test_ids, predict_test, write_submission
from .models import build_linear_probing_head, load_dinov2_backbone
from .training import fit


SOLUTION_REGISTRY = {}


def register_solution(name):
    """Register a solution class under a short name."""

    def decorator(solution_cls):
        SOLUTION_REGISTRY[name.lower()] = solution_cls
        return solution_cls

    return decorator


class BaseSolution:
    """Base class for future solution variants."""

    def __init__(self, config=None):
        self.config = {
            "train_path": "train.h5",
            "val_path": "val.h5",
            "test_path": "test.h5",
            "output_csv": "baseline.csv",
            "batch_size": 16,
            "resize": (98, 98),
            "num_epochs": 100,
            "patience": 10,
            "lr": 0.001,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        if config is not None:
            self.config.update(config)

        self.device = torch.device(self.config["device"])
        self.preprocessing = transforms.Resize(self.config["resize"])
        self.feature_extractor = None
        self.linear_probing = None
        self.history = None

    def fit(self):
        raise NotImplementedError

    def predict_test(self, output_csv="baseline.csv"):
        raise NotImplementedError


@register_solution("baseline")
class BaselineSolution(BaseSolution):
    """Notebook-style baseline that extracts DINOv2 features and trains a linear probe."""

    def _build_dataloaders(self):
        train_dataset = BaselineDataset(self.config["train_path"], self.preprocessing, "train")
        val_dataset = BaselineDataset(self.config["val_path"], self.preprocessing, "train")

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.config["batch_size"])
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=self.config["batch_size"])
        return train_dataloader, val_dataloader

    def _build_model(self):
        self.feature_extractor = load_dinov2_backbone().to(self.device)
        self.feature_extractor.eval()
        self.linear_probing = build_linear_probing_head(self.feature_extractor).to(self.device)

    def fit(self):
        train_dataloader, val_dataloader = self._build_dataloaders()
        self._build_model()

        train_dataset = PrecomputedDataset(*precompute(train_dataloader, self.feature_extractor, self.device))
        val_dataset = PrecomputedDataset(*precompute(val_dataloader, self.feature_extractor, self.device))

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.config["batch_size"])
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=self.config["batch_size"])

        optimizer = torch.optim.Adam(self.linear_probing.parameters(), lr=self.config["lr"])
        criterion = torch.nn.BCELoss()
        self.history = fit(
            train_dataloader,
            val_dataloader,
            self.linear_probing,
            optimizer,
            criterion,
            self.device,
            num_epochs=self.config["num_epochs"],
            patience=self.config["patience"],
            checkpoint_path="best_model.pth",
        )
        return self

    def predict_test(self, output_csv="baseline.csv"):
        if self.feature_extractor is None or self.linear_probing is None:
            self._build_model()
            try:
                self.linear_probing.load_state_dict(torch.load("best_model.pth", weights_only=True))
            except TypeError:
                self.linear_probing.load_state_dict(torch.load("best_model.pth"))

        test_ids = load_test_ids(self.config["test_path"])
        predictions = predict_test(
            test_ids,
            self.preprocessing,
            self.feature_extractor,
            self.linear_probing,
            self.device,
            test_images_path=self.config["test_path"],
        )
        return write_submission(test_ids, predictions, output_csv)


def get_solution(name, config=None):
    """Create a solution by name."""

    try:
        solution_cls = SOLUTION_REGISTRY[name.lower()]
    except KeyError as error:
        available = ", ".join(sorted(SOLUTION_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown solution '{name}'. Available solutions: {available}") from error
    return solution_cls(config)
