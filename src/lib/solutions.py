"""Solution registry and baseline orchestration."""

from __future__ import annotations

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import h5py
import numpy as np

from .augmentations import ReinhardNormalizer, RandomStainColorJitter
from .datasets import BaselineDataset, CenterBalancedBatchSampler, PrecomputedDataset, precompute
from .inference import load_test_ids, predict_test, predict_test_tta, write_submission
from .models import build_linear_probing_head, build_lora_backbone, load_backbone
from .training import fit, fit_frozen_backbone, fit_trainable_backbone, fit_trainable_backbone_with_class_conditional_coral


SOLUTION_REGISTRY = {}


class ToFloat32:
    def __call__(self, x):
        return x.float()


class Clamp01:
    def __call__(self, x):
        return x.clamp(0.0, 1.0)


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
            "backbone_name": "dinov2_vits14",
            "backbone_pretrained": True,
            "backbone_kwargs": {},
            "head_hidden_dim": None,
            "head_dropout": 0.0,
            "head_use_sigmoid": True,
            "fast_aug": False,
            "use_reinhard": False,
            "use_stain_jitter": True,
            "use_color_jitter": True,
            "use_rotations": True,
            "use_amp": False,
            "log_batch_timing": False,
            "prefetch_factor": 2,
            "persistent_workers": True,
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
            ToFloat32(),
            transforms.Resize(self.config["resize"]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.feature_extractor = None
        self.linear_probing = None
        self.history = None
        self.tta_augmentations = []
        self._reinhard_normalizer = None

    def _backbone_name(self):
        return self.config.get("backbone_name", "dinov2_vits14")

    def _backbone_pretrained(self):
        return self.config.get("backbone_pretrained", True)

    def _backbone_kwargs(self):
        return dict(self.config.get("backbone_kwargs", {}))

    def _float32_transform(self):
        return ToFloat32()

    def _resize_transform(self, size=None):
        return transforms.Resize(size or self.config["resize"])

    def _normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _clamp_transform(self):
        return Clamp01()

    def _load_reinhard_normalizer(self):
        if not self.config.get("use_reinhard", False):
            return None

        if self._reinhard_normalizer is not None:
            return self._reinhard_normalizer

        with h5py.File(self.config["train_path"], "r") as hdf:
            first_id = list(hdf.keys())[0]
            reference = torch.tensor(np.array(hdf.get(first_id).get("img"))).float()
        self._reinhard_normalizer = ReinhardNormalizer(reference)
        return self._reinhard_normalizer

    def _build_standard_preprocessing(self, size=None, include_reinhard=False):
        preprocessors = [self._float32_transform(), self._resize_transform(size)]
        if include_reinhard:
            reinhard = self._load_reinhard_normalizer()
            if reinhard is not None:
                preprocessors.append(reinhard)
        preprocessors.append(self._normalize_transform())
        return transforms.Compose(preprocessors)

    def _build_targeted_preprocessing(self, include_reinhard=False, train=True):
        preprocessors = [self._float32_transform(), self._resize_transform()]
        reinhard = self._load_reinhard_normalizer() if include_reinhard else None
        if reinhard is not None:
            preprocessors.append(reinhard)

        if train:
            fast_aug = self.config.get("fast_aug", False)
            if not fast_aug and self.config.get("use_stain_jitter", True):
                preprocessors.append(RandomStainColorJitter(sigma=self.config.get("stain_sigma", 0.1)))
            preprocessors.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
            if not fast_aug and self.config.get("use_rotations", True):
                preprocessors.append(
                    transforms.RandomChoice(
                        [
                            transforms.RandomRotation([0, 0]),
                            transforms.RandomRotation([90, 90]),
                            transforms.RandomRotation([180, 180]),
                            transforms.RandomRotation([270, 270]),
                        ]
                    )
                )
            if self.config.get("use_color_jitter", True):
                preprocessors.append(
                    transforms.ColorJitter(
                        brightness=self.config.get("jitter_brightness", 0.15),
                        contrast=self.config.get("jitter_contrast", 0.15),
                    )
                )
            preprocessors.extend([self._clamp_transform(), self._normalize_transform()])
        else:
            preprocessors.append(self._normalize_transform())

        return transforms.Compose(preprocessors)

    def _build_tta_augmentations(self, include_reinhard=False):
        base = [self._float32_transform(), self._resize_transform()]
        reinhard = self._load_reinhard_normalizer() if include_reinhard else None
        if reinhard is not None:
            base.append(reinhard)

        augmentations = [
            transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0), self._normalize_transform()]),
            transforms.Compose(base + [transforms.RandomVerticalFlip(p=1.0), self._normalize_transform()]),
        ]
        if not self.config.get("fast_aug", False) and self.config.get("use_rotations", True):
            augmentations.extend(
                [
                    transforms.Compose(base + [transforms.RandomRotation([90, 90]), self._normalize_transform()]),
                    transforms.Compose(base + [transforms.RandomRotation([180, 180]), self._normalize_transform()]),
                    transforms.Compose(base + [transforms.RandomRotation([270, 270]), self._normalize_transform()]),
                ]
            )
        if not self.config.get("fast_aug", False) and self.config.get("use_stain_jitter", True):
            augmentations.append(
                transforms.Compose(
                    base
                    + [
                        RandomStainColorJitter(sigma=self.config.get("stain_sigma", 0.1), p=1.0),
                        self._normalize_transform(),
                    ]
                )
            )
        if self.config.get("use_color_jitter", True):
            augmentations.append(
                transforms.Compose(
                    base
                    + [
                        transforms.ColorJitter(
                            brightness=self.config.get("jitter_brightness", 0.15),
                            contrast=self.config.get("jitter_contrast", 0.15),
                        ),
                        self._clamp_transform(),
                        self._normalize_transform(),
                    ]
                )
            )
        return augmentations

    def _make_dataloader(self, dataset, shuffle=False, batch_size=None, batch_sampler=None):
        num_workers = self.config.get("num_workers", 4)
        dataloader_kwargs = {
            "pin_memory": self.device.type == "cuda",
            "drop_last": False,
        }

        if batch_sampler is not None:
            dataloader_kwargs["batch_sampler"] = batch_sampler
        else:
            dataloader_kwargs["shuffle"] = shuffle
            dataloader_kwargs["batch_size"] = batch_size or self.config["batch_size"]

        dataloader_kwargs["num_workers"] = num_workers
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.config.get("persistent_workers", True)
            dataloader_kwargs["prefetch_factor"] = self.config.get("prefetch_factor", 2)

        return DataLoader(dataset, **dataloader_kwargs)

    def _build_frozen_backbone(self):
        backbone = load_backbone(
            self._backbone_name(),
            pretrained=self._backbone_pretrained(),
            **self._backbone_kwargs(),
        ).to(self.device)
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        return backbone

    def _build_trainable_backbone(self):
        return build_lora_backbone(
            self._backbone_name(),
            rank=self.config["lora_rank"],
            alpha=self.config["lora_alpha"],
            pretrained=self._backbone_pretrained(),
            **self._backbone_kwargs(),
        ).to(self.device)

    def _build_linear_head(self):
        return build_linear_probing_head(
            self.feature_extractor,
            hidden_dim=self.config.get("head_hidden_dim"),
            dropout=self.config.get("head_dropout", 0.0),
            use_sigmoid=self.config.get("head_use_sigmoid", True),
        ).to(self.device)

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

        train_dataloader = self._make_dataloader(train_dataset, shuffle=True)
        val_dataloader = self._make_dataloader(val_dataset, shuffle=False)
        return train_dataloader, val_dataloader

    def _build_model(self):
        self.feature_extractor = self._build_frozen_backbone()
        self.linear_probing = self._build_linear_head()

    def fit(self):
        train_dataloader, val_dataloader = self._build_dataloaders()
        self._build_model()

        train_dataset = PrecomputedDataset(*precompute(train_dataloader, self.feature_extractor, self.device))
        val_dataset = PrecomputedDataset(*precompute(val_dataloader, self.feature_extractor, self.device))

        train_dataloader = self._make_dataloader(train_dataset, shuffle=True)
        val_dataloader = self._make_dataloader(val_dataset, shuffle=False)

        optimizer = torch.optim.Adam(self.linear_probing.parameters(), lr=self.config["head_lr"])
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

@register_solution("baseline_224")
class Baseline224Solution(BaseSolution):
    """
    Baseline with higher resolution (224x224).
    Everything else is strictly identical.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.config["resize"] = (224, 224)
        self.preprocessing = self._build_standard_preprocessing(size=self.config["resize"])

    def _build_dataloaders(self):
        train_dataset = BaselineDataset(self.config["train_path"], self.preprocessing, "train")
        val_dataset = BaselineDataset(self.config["val_path"], self.preprocessing, "train")

        train_dataloader = self._make_dataloader(train_dataset, shuffle=True)
        val_dataloader = self._make_dataloader(val_dataset, shuffle=False)
        return train_dataloader, val_dataloader

    def _build_model(self):
        self.feature_extractor = self._build_frozen_backbone()
        self.linear_probing = self._build_linear_head()

    def fit(self):
        train_dataloader, val_dataloader = self._build_dataloaders()
        self._build_model()

        train_dataset = PrecomputedDataset(*precompute(train_dataloader, self.feature_extractor, self.device))
        val_dataset = PrecomputedDataset(*precompute(val_dataloader, self.feature_extractor, self.device))

        train_dataloader = self._make_dataloader(train_dataset, shuffle=True)
        val_dataloader = self._make_dataloader(val_dataset, shuffle=False)

        optimizer = torch.optim.Adam(self.linear_probing.parameters(), lr=self.config["head_lr"])
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
            checkpoint_path=self.config.get("checkpoint_path", "best_model_224.pth"),
        )
        return self

    def predict_test(self, output_csv="baseline_224.csv"):
        if self.feature_extractor is None or self.linear_probing is None:
            self._build_model()
            checkpoint_path = self.config.get("checkpoint_path", "best_model_224.pth")
            try:
                self.linear_probing.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            except TypeError:
                self.linear_probing.load_state_dict(torch.load(checkpoint_path))

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


@register_solution("baseline_color_jitter")
class BaselineColorJitterSolution(BaseSolution):
    """
    Baseline + train-only color jitter. Augmentations are applied on every batch,
    so features are recomputed each epoch through the frozen backbone.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.train_preprocessing = transforms.Compose(
            [
                self._float32_transform(),
                self._resize_transform(),
                transforms.ColorJitter(
                    brightness=self.config.get("jitter_brightness", 0.15),
                    contrast=self.config.get("jitter_contrast", 0.15),
                    saturation=self.config.get("jitter_saturation", 0.15),
                    hue=self.config.get("jitter_hue", 0.02),
                ),
                self._clamp_transform(),
                self._normalize_transform(),
            ]
        )
        self.eval_preprocessing = self._build_standard_preprocessing(size=self.config["resize"])

    def _build_dataloaders(self):
        train_dataset = BaselineDataset(self.config["train_path"], self.train_preprocessing, "train")
        val_dataset = BaselineDataset(self.config["val_path"], self.eval_preprocessing, "train")

        train_dataloader = self._make_dataloader(train_dataset, shuffle=True)
        val_dataloader = self._make_dataloader(val_dataset, shuffle=False)
        return train_dataloader, val_dataloader

    def _build_model(self):
        self.feature_extractor = self._build_frozen_backbone()
        self.linear_probing = self._build_linear_head()

    def fit(self):
        train_dataloader, val_dataloader = self._build_dataloaders()
        self._build_model()
        optimizer = torch.optim.Adam(self.linear_probing.parameters(), lr=self.config["head_lr"])
        criterion = torch.nn.BCELoss()
        self.history = fit_frozen_backbone(
            train_dataloader,
            val_dataloader,
            self.feature_extractor,
            self.linear_probing,
            optimizer,
            criterion,
            self.device,
            num_epochs=self.config["num_epochs"],
            patience=self.config["patience"],
            checkpoint_path=self.config.get("checkpoint_path", "best_model_color_jitter.pth"),
            use_amp=self.config.get("use_amp", self.device.type == "cuda"),
            log_batch_timing=self.config.get("log_batch_timing", False),
        )
        return self

    def predict_test(self, output_csv="baseline_color_jitter.csv"):
        if self.feature_extractor is None or self.linear_probing is None:
            self._build_model()
            checkpoint_path = self.config.get("checkpoint_path", "best_model_color_jitter.pth")
            try:
                self.linear_probing.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            except (TypeError, RuntimeError):
                state_dict = torch.load(checkpoint_path)
                try:
                    self.linear_probing.load_state_dict(state_dict)
                except RuntimeError:
                    full_model = torch.nn.Sequential(self.feature_extractor, self.linear_probing)
                    full_model.load_state_dict(state_dict)
            self.feature_extractor.eval()
            self.linear_probing.eval()

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


@register_solution("baseline_224_targeted_augmentations")
class Baseline224TargetedAugmentationsSolution(BaseSolution):
    """Baseline 224x224 with targeted augmentations applied every batch through the frozen backbone."""

    def __init__(self, config=None):
        super().__init__(config)
        self.config["resize"] = (224, 224)
        self.train_preprocessing = self._build_targeted_preprocessing(include_reinhard=self.config.get("use_reinhard", False), train=True)
        self.eval_preprocessing = self._build_targeted_preprocessing(include_reinhard=self.config.get("use_reinhard", False), train=False)

    def _build_dataloaders(self):
        train_dataset = BaselineDataset(self.config["train_path"], self.train_preprocessing, "train")
        val_dataset = BaselineDataset(self.config["val_path"], self.eval_preprocessing, "train")

        train_dataloader = self._make_dataloader(train_dataset, shuffle=True)
        val_dataloader = self._make_dataloader(val_dataset, shuffle=False)
        return train_dataloader, val_dataloader

    def _build_model(self):
        self.feature_extractor = self._build_frozen_backbone()
        self.linear_probing = self._build_linear_head()

    def fit(self):
        train_dataloader, val_dataloader = self._build_dataloaders()
        self._build_model()
        optimizer = torch.optim.Adam(self.linear_probing.parameters(), lr=self.config["head_lr"])
        criterion = torch.nn.BCELoss()
        self.history = fit_frozen_backbone(
            train_dataloader,
            val_dataloader,
            self.feature_extractor,
            self.linear_probing,
            optimizer,
            criterion,
            self.device,
            num_epochs=self.config["num_epochs"],
            patience=self.config["patience"],
            checkpoint_path=self.config.get("checkpoint_path", "best_model_224_targeted_augmentations.pth"),
            use_amp=self.config.get("use_amp", self.device.type == "cuda"),
            log_batch_timing=self.config.get("log_batch_timing", False),
        )
        return self

    def predict_test(self, output_csv="baseline_224_targeted_augmentations.csv"):
        if self.feature_extractor is None or self.linear_probing is None:
            self._build_model()
            checkpoint_path = self.config.get("checkpoint_path", "best_model_224_targeted_augmentations.pth")
            try:
                self.linear_probing.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            except (TypeError, RuntimeError):
                state_dict = torch.load(checkpoint_path)
                try:
                    self.linear_probing.load_state_dict(state_dict)
                except RuntimeError:
                    full_model = torch.nn.Sequential(self.feature_extractor, self.linear_probing)
                    full_model.load_state_dict(state_dict)
            self.feature_extractor.eval()
            self.linear_probing.eval()

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
class LoRATargetedAugmentationsSolution(BaseSolution):
    """LoRA fine-tuning of a ViT backbone at 224x224 with targeted augmentations on training set."""

    def __init__(self, config=None):
        super().__init__(config)
        self.config["resize"] = (224, 224)
        self.config.setdefault("use_amp", self.device.type == "cuda")
        self.train_preprocessing = self._build_targeted_preprocessing(include_reinhard=self.config.get("use_reinhard", False), train=True)
        self.eval_preprocessing = self._build_targeted_preprocessing(include_reinhard=self.config.get("use_reinhard", False), train=False)
        self.tta_augmentations = self._build_tta_augmentations(include_reinhard=self.config.get("use_reinhard", False))

    def _build_dataloaders(self):
        train_dataset = BaselineDataset(self.config["train_path"], self.train_preprocessing, "train")
        val_dataset = BaselineDataset(self.config["val_path"], self.eval_preprocessing, "train")

        train_dataloader = self._make_dataloader(train_dataset, shuffle=True)
        val_dataloader = self._make_dataloader(val_dataset, shuffle=False)
        return train_dataloader, val_dataloader

    def _build_model(self):
        self.feature_extractor = self._build_trainable_backbone()
        self.linear_probing = self._build_linear_head()

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

    def fit(self):
        train_dataloader, val_dataloader = self._build_dataloaders()
        self._build_model()

        optimizer = torch.optim.Adam([
            {"params": [p for p in self.feature_extractor.parameters() if p.requires_grad], "lr": self.config["backbone_lr"]},
            {"params": self.linear_probing.parameters(), "lr": self.config["head_lr"]},
        ])
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
            use_amp=self.config.get("use_amp", self.device.type == "cuda"),
            log_batch_timing=self.config.get("log_batch_timing", False),
        )
        return self

    def predict_test(self, output_csv="lora_dinov2_targeted_augmentations.csv"):
        if self.feature_extractor is None or self.linear_probing is None:
            self._build_model()
            checkpoint_path = self.config.get("checkpoint_path", "best_model_lora_dinov2_targeted_augmentations.pth")
            ckpt = torch.load(checkpoint_path, weights_only=True)
            self.feature_extractor.load_state_dict(ckpt["lora"], strict=False)
            self.linear_probing.load_state_dict(ckpt["head"])
            self.feature_extractor.eval()
            self.linear_probing.eval()

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

    def predict_test_tta(self, output_csv="lora_dinov2_targeted_augmentations_tta.csv"):
        if self.feature_extractor is None or self.linear_probing is None:
            self._build_model()
            checkpoint_path = self.config.get("checkpoint_path", "best_model_lora_dinov2_targeted_augmentations.pth")
            ckpt = torch.load(checkpoint_path, weights_only=True)
            self.feature_extractor.load_state_dict(ckpt["lora"], strict=False)
            self.linear_probing.load_state_dict(ckpt["head"])
            self.feature_extractor.eval()
            self.linear_probing.eval()

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
class LoRAClassConditionalCoralSolution(BaseSolution):
    """LoRA fine-tuning with class-conditional CORAL across training centers."""

    def __init__(self, config=None):
        super().__init__(config)
        self.config["resize"] = (224, 224)
        self.config.setdefault("use_amp", self.device.type == "cuda")
        self.train_preprocessing = self._build_targeted_preprocessing(include_reinhard=False, train=True)
        self.eval_preprocessing = self._build_targeted_preprocessing(include_reinhard=False, train=False)

    def _build_dataloaders(self):
        train_dataset = BaselineDataset(self.config["train_path"], self.preprocessing, "train", return_center=True)
        val_dataset = BaselineDataset(self.config["val_path"], self.preprocessing, "train")

        sampler = CenterBalancedBatchSampler(train_dataset.centers, self.config["batch_size"])
        train_dataloader = self._make_dataloader(train_dataset, batch_sampler=sampler)
        val_dataloader = self._make_dataloader(val_dataset, shuffle=False)
        return train_dataloader, val_dataloader

    def _build_model(self):
        self.feature_extractor = self._build_trainable_backbone()
        self.linear_probing = self._build_linear_head()

    def fit(self):
        train_dataloader, val_dataloader = self._build_dataloaders()
        self._build_model()

        optimizer = torch.optim.Adam([
            {"params": [p for p in self.feature_extractor.parameters() if p.requires_grad], "lr": self.config["backbone_lr"]},
            {"params": self.linear_probing.parameters(), "lr": self.config["head_lr"]},
        ])
        criterion = torch.nn.BCELoss()
        self.history = fit_trainable_backbone_with_class_conditional_coral(
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
            use_amp=self.config.get("use_amp", self.device.type == "cuda"),
            log_batch_timing=self.config.get("log_batch_timing", False),
        )
        return self

    def predict_test(self, output_csv="lora_dinov2_class_conditional_coral.csv"):
        if self.feature_extractor is None or self.linear_probing is None:
            self._build_model()
            checkpoint_path = self.config.get("checkpoint_path", "best_model_lora_dinov2_class_conditional_coral.pth")
            ckpt = torch.load(checkpoint_path, weights_only=True)
            self.feature_extractor.load_state_dict(ckpt["lora"], strict=False)
            self.linear_probing.load_state_dict(ckpt["head"])
            self.feature_extractor.eval()
            self.linear_probing.eval()

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


@register_solution("lora_uni_targeted_augmentations")
class LoRAUNITargetedAugmentationsSolution(LoRATargetedAugmentationsSolution):
    """UNI-targeted augmentation run using the shared LoRA ViT pipeline."""

    def __init__(self, config=None):
        super().__init__(config)
        self.config["backbone_name"] = "uni"
        self.config.setdefault("output_csv", "uni_targeted_augmentations.csv")
        self.config.setdefault("checkpoint_path", "best_model_lora_uni_targeted_augmentations.pth")


def get_solution(name, config=None):
    """Create a solution by name."""

    try:
        solution_cls = SOLUTION_REGISTRY[name.lower()]
    except KeyError as error:
        available = ", ".join(sorted(SOLUTION_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown solution '{name}'. Available solutions: {available}") from error
    return solution_cls(config)
