"""Run test-only inference with the UNI stage-1 checkpoint."""

from __future__ import annotations

import argparse

from src.main import get_solution

import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run test inference for UNI stage-1 checkpoint.")
    parser.add_argument("--test-path", default="test.h5")
    parser.add_argument("--output-csv", default="uni_stage1_test_predictions.csv")
    parser.add_argument("--checkpoint-path", default="best_model_uni_stage1.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-reinhard", action="store_true", help="Disable Reinhard normalization at test time.")
    return parser


def build_config(args: argparse.Namespace) -> dict:
    config = {
        "test_path": args.test_path,
        "checkpoint_path": args.checkpoint_path,
        "batch_size": args.batch_size,
        "backbone_name": "uni",
        "resize": (224, 224),
        "head_type": "mlp",
        "head_hidden_dim": 1024,
        "head_dropout": 0.1,
        "use_reinhard": not args.no_reinhard,
        "fast_aug": False,
        "backbone_kwargs": {
            "model_name": "hf-hub:MahmoodLab/UNI",
            "init_values": 1e-5,
            "dynamic_img_size": True,
        },
    }
    if args.device is not None:
        config["device"] = args.device
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = build_config(args)

    print("Running test-only inference with UNI stage-1 checkpoint")
    print(f"  checkpoint_path={config['checkpoint_path']}")
    print(f"  test_path={config['test_path']}")
    print(f"  output_csv={args.output_csv}")

    solution = get_solution("baseline_224_targeted_augmentations", config)
    solution.predict_test(output_csv=args.output_csv)

    print("Done.")


if __name__ == "__main__":
    main()