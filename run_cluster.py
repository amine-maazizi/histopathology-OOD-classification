"""Cluster entrypoint for the 224 targeted-augmentation baseline.

This script mirrors the notebook section:
- "Stain Jitters + Higher resolution"

It trains `baseline_224_targeted_augmentations` and writes test predictions.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from src.main import get_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline_224_targeted_augmentations training and inference."
    )

    parser.add_argument("--train-path", default="train.h5", help="Path to training .h5 file.")
    parser.add_argument("--val-path", default="val.h5", help="Path to validation .h5 file.")
    parser.add_argument("--test-path", default="test.h5", help="Path to test .h5 file.")

    parser.add_argument(
        "--output-csv",
        default="baseline_224_targeted_augmentations.csv",
        help="Submission CSV output path.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="best_model_224_targeted_augmentations.pth",
        help="Checkpoint path for best model weights.",
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--stain-sigma", type=float, default=0.1)
    parser.add_argument("--jitter-brightness", type=float, default=0.15)
    parser.add_argument("--jitter-contrast", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' selects CUDA if available.",
    )

    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training and only run test prediction using checkpoint.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = {
        "train_path": args.train_path,
        "val_path": args.val_path,
        "test_path": args.test_path,
        "output_csv": args.output_csv,
        "batch_size": args.batch_size,
        "resize": (224, 224),
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "checkpoint_path": args.checkpoint_path,
        "stain_sigma": args.stain_sigma,
        "jitter_brightness": args.jitter_brightness,
        "jitter_contrast": args.jitter_contrast,
        "num_workers": args.num_workers,
        "device": resolve_device(args.device),
    }

    solution = get_solution("baseline_224_targeted_augmentations", config)

    if not args.predict_only:
        solution.fit()

    solution.predict_test(output_csv=args.output_csv)


if __name__ == "__main__":
    main()
