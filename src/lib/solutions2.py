"""Solution registry and baseline orchestration."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import h5py
import numpy as np

from .augmentations import ReinhardNormalizer, StainColorJitter, RandomStainColorJitter
from .datasets import BaselineDataset, PrecomputedDataset, precompute
from .inference import load_test_ids, predict_test, predict_test_tta, write_submission
from .datasets import BaselineDataset, CenterBalancedBatchSampler, PrecomputedDataset, precompute
from .models import build_linear_probing_head, build_lora_dinov2, load_dinov2_backbone
from .training import fit_frozen_backbone, fit_trainable_backbone


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
            "head_lr": 0.001,
            "backbone_lr": 4e-4,
            "checkpoint_path": "best_model.pth",
            "jitter_brightness": 0.15,
            "jitter_contrast": 0.15,
            "jitter_saturation": 0.15,
            "jitter_hue": 0.02,
            "stain_sigma": 0.1, 
            "lora_rank": 8,
            "lora_alpha": 1.0,
            "coral_lambda": 0.05,
            "coral_eps": 1e-5,
            "num_workers": 4,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        if config is not None:
            self.config.update(config)

        self.device = torch.device(self.config["device"])
        self.preprocessing = transforms.Compose([
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize(self.config["resize"]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.feature_extractor = None
        self.linear_probing = None
        self.history = None
        self.tta_augmentations = []

    def fit(self):
        raise NotImplementedError

    def predict_test(self, output_csv="baseline.csv"):
        raise NotImplementedError


class FrozenBackboneSolution(BaseSolution):
    precompute = False

    def __init__(self, config=None):
        super().__init__(config)
        self.train_preprocessing = self.preprocessing
        self.eval_preprocessing = self.preprocessing

    def _build_model(self):
        self.feature_extractor = load_dinov2_backbone().to(self.device)
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)
        self.linear_probing = build_linear_probing_head(self.feature_extractor).to(self.device)

    def _make_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.config["batch_size"],
            num_workers=self.config.get("num_workers", 4),
            pin_memory=self.device.type == "cuda",
        )

    def _build_dataloaders(self):
        train_dataset = BaselineDataset(self.config["train_path"], self.train_preprocessing, "train")
        val_dataset = BaselineDataset(self.config["val_path"], self.eval_preprocessing, "train")
        return self._make_dataloader(train_dataset, shuffle=True), self._make_dataloader(val_dataset, shuffle=False)

    def _load_checkpoint(self, path):
        try:
            self.linear_probing.load_state_dict(torch.load(path, weights_only=True))
        except (TypeError, RuntimeError):
            state_dict = torch.load(path)
            try:
                self.linear_probing.load_state_dict(state_dict)
            except RuntimeError:
                torch.nn.Sequential(self.feature_extractor, self.linear_probing).load_state_dict(state_dict)
        self.feature_extractor.eval()
        self.linear_probing.eval()

    def fit(self):
        train_dataloader, val_dataloader = self._build_dataloaders()
        self._build_model()
        optimizer = torch.optim.Adam(self.linear_probing.parameters(), lr=self.config["head_lr"])
        criterion = torch.nn.BCELoss()

        if self.precompute:
            train_dataset = PrecomputedDataset(*precompute(train_dataloader, self.feature_extractor, self.device))
            val_dataset = PrecomputedDataset(*precompute(val_dataloader, self.feature_extractor, self.device))
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.config["batch_size"])
            val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=self.config["batch_size"])

        self.history = fit_frozen_backbone(
            train_dataloader,
            val_dataloader,
            self.linear_probing,
            optimizer,
            criterion,
            self.device,
            num_epochs=self.config["num_epochs"],
            patience=self.config["patience"],
            checkpoint_path=self.config.get("checkpoint_path", "best_model.pth"),
            feature_extractor=None if self.precompute else self.feature_extractor,
        )
        return self

    def predict_test(self, output_csv=None):
        if output_csv is None:
            output_csv = self.config.get("output_csv", "submission.csv")
        if self.feature_extractor is None or self.linear_probing is None:
            self._build_model()
            self._load_checkpoint(self.config.get("checkpoint_path", "best_model.pth"))
        test_ids = load_test_ids(self.config["test_path"])
        predictions = predict_test(
            test_ids,
            self.eval_preprocessing,
            self.feature_extractor,
            self.linear_probing,
            self.device,
            test_images_path=self.config["test_path"],
        )
        return write_submission(test_ids, predictions, output_csv)


@register_solution("baseline")
class BaselineSolution(FrozenBackboneSolution):
    precompute = True


@register_solution("baseline_color_jitter")
class BaselineColorJitterSolution(FrozenBackboneSolution):
    def __init__(self, config=None):
        super().__init__(config)
        
        if config is None or "checkpoint_path" not in config:
            self.config["checkpoint_path"] = "best_model_color_jitter.pth"
        
        self.train_preprocessing = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.float()),
                transforms.Resize(self.config["resize"]),
                transforms.ColorJitter(
                    brightness=self.config.get("jitter_brightness", 0.15),
                    contrast=self.config.get("jitter_contrast", 0.15),
                    saturation=self.config.get("jitter_saturation", 0.15),
                    hue=self.config.get("jitter_hue", 0.02),
                ),
                transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.eval_preprocessing = transforms.Compose([
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize(self.config["resize"]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


@register_solution("baseline_224_targeted_augmentations")
class Baseline224TargetedAugmentationsSolution(FrozenBackboneSolution):
    def __init__(self, config=None):
        super().__init__(config)

        if config is None or "checkpoint_path" not in config:
            self.config["checkpoint_path"] = "best_model_224_targeted_augmentations.pth"
        
        self.config["resize"] = (224, 224)
        resize = transforms.Resize(self.config["resize"])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        reinhard = None
        if self.config.get("use_reinhard", False):
            with h5py.File(self.config["train_path"], "r") as hdf:
                first_id = list(hdf.keys())[0]
                reference = torch.tensor(np.array(hdf.get(first_id).get("img"))).float()
            reinhard = ReinhardNormalizer(reference)

        train_preprocessing = [resize]
        if reinhard is not None:
            train_preprocessing.append(reinhard)
        train_preprocessing += [
            transforms.Lambda(lambda x: x.float()),
            RandomStainColorJitter(sigma=self.config.get("stain_sigma", 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation([0, 0]),
                transforms.RandomRotation([90, 90]),
                transforms.RandomRotation([180, 180]),
                transforms.RandomRotation([270, 270]),
            ]),
            transforms.ColorJitter(
                brightness=self.config.get("jitter_brightness", 0.15),
                contrast=self.config.get("jitter_contrast", 0.15),
            ),
            transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
            normalize,
        ]
        self.train_preprocessing = transforms.Compose(train_preprocessing)

        eval_preprocessing = [transforms.Lambda(lambda x: x.float()), resize]
        if reinhard is not None:
            eval_preprocessing.append(reinhard)
        eval_preprocessing.append(normalize)
        self.eval_preprocessing = transforms.Compose(eval_preprocessing)


class TrainableBackboneSolution(BaseSolution):

    def __init__(self, config=None):
        super().__init__(config)
        self.config["resize"] = (224, 224)
        resize = transforms.Resize(self.config["resize"])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        reinhard = None
        if self.config.get("use_reinhard", False):
            with h5py.File(self.config["train_path"], "r") as hdf:
                first_id = list(hdf.keys())[0]
                ref_img = torch.tensor(np.array(hdf.get(first_id).get("img"))).float()
            reinhard = ReinhardNormalizer(ref_img)

        train_preprocessing = [transforms.Lambda(lambda x: x.float()), resize]
        if reinhard is not None:
            train_preprocessing.append(reinhard)
        train_preprocessing += [
            RandomStainColorJitter(sigma=self.config.get("stain_sigma", 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation([0, 0]),
                transforms.RandomRotation([90, 90]),
                transforms.RandomRotation([180, 180]),
                transforms.RandomRotation([270, 270]),
            ]),
            transforms.ColorJitter(
                brightness=self.config.get("jitter_brightness", 0.15),
                contrast=self.config.get("jitter_contrast", 0.15),
            ),
            transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
            normalize,
        ]
        self.train_preprocessing = transforms.Compose(train_preprocessing)

        eval_preprocessing = [transforms.Lambda(lambda x: x.float()), resize]
        if reinhard is not None:
            eval_preprocessing.append(reinhard)
        eval_preprocessing.append(normalize)
        self.eval_preprocessing = transforms.Compose(eval_preprocessing)

    def _build_model(self):
        self.feature_extractor = build_lora_dinov2(
            rank=self.config["lora_rank"],
            alpha=self.config["lora_alpha"],
        ).to(self.device)
        self.linear_probing = build_linear_probing_head(self.feature_extractor).to(self.device)

        head_init_checkpoint = self.config.get("head_init_checkpoint")
        if head_init_checkpoint:
            try:
                state_dict = torch.load(head_init_checkpoint, weights_only=True)
            except TypeError:
                state_dict = torch.load(head_init_checkpoint)
            try:
                self.linear_probing.load_state_dict(state_dict)
            except Exception as error:
                raise RuntimeError(
                    "Failed to initialize linear head from "
                    f"'{head_init_checkpoint}'. Expected a plain linear probe state_dict with matching shapes."
                ) from error

    def _make_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.config["batch_size"],
            num_workers=self.config.get("num_workers", 4),
            pin_memory=self.device.type == "cuda",
        )

    def _build_dataloaders(self):
        train_dataset = BaselineDataset(self.config["train_path"], self.train_preprocessing, "train")
        val_dataset = BaselineDataset(self.config["val_path"], self.eval_preprocessing, "train")
        return self._make_dataloader(train_dataset, shuffle=True), self._make_dataloader(val_dataset, shuffle=False)

    def _build_optimizer(self):
        return torch.optim.Adam([
            {"params": [p for p in self.feature_extractor.parameters() if p.requires_grad], "lr": self.config["backbone_lr"]},
            {"params": self.linear_probing.parameters(), "lr": self.config["head_lr"]},
        ])

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.feature_extractor.load_state_dict(ckpt["lora"], strict=False)
        self.linear_probing.load_state_dict(ckpt["head"])
        self.feature_extractor.eval()
        self.linear_probing.eval()

    def predict_test(self, output_csv=None):
        if output_csv is None:
            output_csv = self.config.get("output_csv", "submission.csv")
        if self.feature_extractor is None or self.linear_probing is None:
            self._build_model()
            self._load_checkpoint(self.config.get("checkpoint_path", "best_model.pth"))
        test_ids = load_test_ids(self.config["test_path"])
        predictions = predict_test(
            test_ids,
            self.eval_preprocessing,
            self.feature_extractor,
            self.linear_probing,
            self.device,
            test_images_path=self.config["test_path"],
        )
        return write_submission(test_ids, predictions, output_csv)


@register_solution("lora_dinov2_targeted_augmentations")
class LoRATargetedAugmentationsSolution(TrainableBackboneSolution):

    def __init__(self, config=None):
        super().__init__(config)
        
        if config is None or "checkpoint_path" not in config:
            self.config["checkpoint_path"] = "best_model_lora_dinov2_targeted_augmentations.pth"
        
        resize = transforms.Resize(self.config["resize"])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        reinhard = None
        if self.config.get("use_reinhard", False):
            with h5py.File(self.config["train_path"], "r") as hdf:
                first_id = list(hdf.keys())[0]
                ref_img = torch.tensor(np.array(hdf.get(first_id).get("img"))).float()
            reinhard = ReinhardNormalizer(ref_img)

        tta_base = [transforms.Lambda(lambda x: x.float()), resize, reinhard] if reinhard is not None else [transforms.Lambda(lambda x: x.float()), resize]
        self.tta_augmentations = [
            transforms.Compose(tta_base + [transforms.RandomHorizontalFlip(p=1.0), normalize]),
            transforms.Compose(tta_base + [transforms.RandomVerticalFlip(p=1.0), normalize]),
            transforms.Compose(tta_base + [transforms.RandomRotation([90, 90]), normalize]),
            transforms.Compose(tta_base + [transforms.RandomRotation([180, 180]), normalize]),
            transforms.Compose(tta_base + [transforms.RandomRotation([270, 270]), normalize]),
            transforms.Compose(tta_base + [RandomStainColorJitter(sigma=self.config.get("stain_sigma", 0.1), p=1.0), normalize]),
            transforms.Compose(tta_base + [transforms.ColorJitter(brightness=self.config.get("jitter_brightness", 0.15), contrast=self.config.get("jitter_contrast", 0.15)), normalize]),
        ]

    def fit(self):
        train_dataloader, val_dataloader = self._build_dataloaders()
        self._build_model()
        optimizer = self._build_optimizer()
        scheduler = None
        if self.config.get("scheduler") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=5, eta_min=1e-5
            )
        criterion = torch.nn.BCELoss()
        self.history = fit_trainable_backbone(
            train_dataloader,
            val_dataloader,
            self.feature_extractor,
            self.linear_probing,
            optimizer,
            criterion,
            self.device,
            num_epochs=self.config["num_epochs"],
            patience=self.config["patience"],
            checkpoint_path=self.config.get("checkpoint_path", "best_model_lora_dinov2_targeted_augmentations.pth"),
            scheduler=scheduler,
        )
        return self

    def predict_test_tta(self, output_csv="lora_dinov2_targeted_augmentations_tta.csv"):
        if self.feature_extractor is None or self.linear_probing is None:
            self._build_model()
            self._load_checkpoint(self.config.get("checkpoint_path", "best_model_lora_dinov2_targeted_augmentations.pth"))
        test_ids = load_test_ids(self.config["test_path"])
        predictions = predict_test_tta(
            test_ids,
            self.eval_preprocessing,
            self.tta_augmentations,
            self.feature_extractor,
            self.linear_probing,
            self.device,
            test_images_path=self.config["test_path"],
        )
        return write_submission(test_ids, predictions, output_csv)


@register_solution("lora_dinov2_class_conditional_coral")
class LoRAClassConditionalCoralSolution(TrainableBackboneSolution):

    def __init__(self, config=None):
        super().__init__(config)
        
        if config is None or "checkpoint_path" not in config:
            self.config["checkpoint_path"] = "best_model_lora_dinov2_class_conditional_coral.pth"

    def _build_dataloaders(self):
        train_dataset = BaselineDataset(self.config["train_path"], self.train_preprocessing, "train", return_center=True)
        val_dataset = BaselineDataset(self.config["val_path"], self.eval_preprocessing, "train")
        sampler = CenterBalancedBatchSampler(train_dataset.centers, self.config["batch_size"])
        num_workers = self.config.get("num_workers", 4)
        pin_memory = self.device.type == "cuda"
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.config["batch_size"],
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_dataloader, val_dataloader

    def fit(self):
        train_dataloader, val_dataloader = self._build_dataloaders()
        self._build_model()
        optimizer = self._build_optimizer()
        criterion = torch.nn.BCELoss()
        self.history = fit_trainable_backbone(
            train_dataloader,
            val_dataloader,
            self.feature_extractor,
            self.linear_probing,
            optimizer,
            criterion,
            self.device,
            coral_lambda=self.config.get("coral_lambda", 0.05),
            coral_eps=self.config.get("coral_eps", 1e-5),
            num_epochs=self.config["num_epochs"],
            patience=self.config["patience"],
            checkpoint_path=self.config.get("checkpoint_path", "best_model_lora_dinov2_class_conditional_coral.pth"),
        )
        return self


def get_solution(name, config=None):
    """Create a solution by name."""

    try:
        solution_cls = SOLUTION_REGISTRY[name.lower()]
    except KeyError as error:
        available = ", ".join(sorted(SOLUTION_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown solution '{name}'. Available solutions: {available}") from error
    return solution_cls(config)