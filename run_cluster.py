"""Standalone runner for the serious UNI LoRA recipe."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.main import get_solution


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

def build_parser():
	parser = argparse.ArgumentParser(description="Run the UNI LoRA histopathology recipe.")
	parser.add_argument("--train-path", default="train.h5")
	parser.add_argument("--val-path", default="val.h5")
	parser.add_argument("--test-path", default="test.h5")
	parser.add_argument("--output-csv", default="uni_targeted_augmentations.csv")
	parser.add_argument("--checkpoint-path", default="best_model_lora_uni_targeted_augmentations.pth")
	parser.add_argument("--head-init-checkpoint", default=None)
	parser.add_argument("--stage1-checkpoint-path", default="best_model_uni_stage1.pth")
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--num-epochs", type=int, default=100)
	parser.add_argument("--patience", type=int, default=10)
	parser.add_argument("--head-lr", type=float, default=1e-3)
	parser.add_argument("--backbone-lr", type=float, default=1e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--warmup-epochs", type=int, default=5)
	parser.add_argument("--lora-rank", type=int, default=4)
	parser.add_argument("--lora-alpha", type=float, default=1.0)
	parser.add_argument("--head-hidden-dim", type=int, default=1024)
	parser.add_argument("--head-dropout", type=float, default=0.1)
	parser.add_argument("--device", default=None)
	parser.add_argument("--fast-aug", action="store_true", help="Disable the expensive targeted augmentations.")
	parser.add_argument("--no-reinhard", action="store_true", help="Disable Reinhard normalization.")
	parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
	return parser


def build_config(args):
	config = {
		"train_path": args.train_path,
		"val_path": args.val_path,
		"test_path": args.test_path,
		"output_csv": args.output_csv,
		"checkpoint_path": args.checkpoint_path,
		"batch_size": args.batch_size,
		"num_epochs": args.num_epochs,
		"patience": args.patience,
		"head_type": "mlp",
		"head_hidden_dim": args.head_hidden_dim,
		"head_dropout": args.head_dropout,
		"head_lr": args.head_lr,
		"backbone_lr": args.backbone_lr,
		"optimizer_name": "adamw",
		"weight_decay": args.weight_decay,
		"scheduler": "cosine",
		"warmup_epochs": args.warmup_epochs,
		"lora_rank": args.lora_rank,
		"lora_alpha": args.lora_alpha,
		"lora_targets": ("qkv",),
		"use_amp": not args.no_amp,
		"use_reinhard": not args.no_reinhard,
		"fast_aug": args.fast_aug,
		"stain_sigma": 0.1,
		"jitter_brightness": 0.15,
		"jitter_contrast": 0.15,
		"backbone_name": "uni",
		"backbone_kwargs": {
			"model_name": "hf-hub:MahmoodLab/UNI",
			"init_values": 1e-5,
			"dynamic_img_size": True,
		},
	}
	if args.device is not None:
		config["device"] = args.device
	return config


def build_stage1_config(args):
	config = build_config(args)
	config.update(
		{
			"backbone_name": "uni",
			"resize": (224, 224),
			"checkpoint_path": args.stage1_checkpoint_path,
			"head_type": "mlp",
			"head_hidden_dim": 1024,
			"head_dropout": 0.1,
			"head_lr": 1e-3,
			"use_reinhard": True,
			"fast_aug": False,
			"backbone_kwargs": {
				"model_name": "hf-hub:MahmoodLab/UNI",
				"init_values": 1e-5,
				"dynamic_img_size": True,
			},
		}
	)
	return config


def build_stage2_config(args, stage1_checkpoint_path):
	config = build_config(args)
	config.update(
		{
			"backbone_name": "uni",
			"resize": (224, 224),
			"checkpoint_path": args.checkpoint_path,
			"head_init_checkpoint": args.head_init_checkpoint or stage1_checkpoint_path,
			"head_type": "mlp",
			"head_hidden_dim": 1024,
			"head_dropout": 0.1,
			"head_lr": 1e-3,
			"backbone_lr": 1e-4,
			"optimizer_name": "adamw",
			"weight_decay": 1e-4,
			"scheduler": "cosine",
			"warmup_epochs": 5,
			"lora_rank": 4,
			"lora_alpha": 1.0,
			"lora_targets": ("qkv",),
			"use_reinhard": True,
			"fast_aug": False,
			"backbone_kwargs": {
				"model_name": "hf-hub:MahmoodLab/UNI",
				"init_values": 1e-5,
				"dynamic_img_size": True,
			},
		}
	)
	return config


def main():
	args = build_parser().parse_args()
	stage1_config = build_stage1_config(args)
	stage2_config = build_stage2_config(args, args.stage1_checkpoint_path)

	print("Running UNI stage 1 (frozen backbone, feature alignment):")
	print(f"  checkpoint_path={stage1_config['checkpoint_path']}")
	print(f"  use_reinhard={stage1_config['use_reinhard']}")
	print(f"  head_type={stage1_config['head_type']}")
	print("Running UNI stage 2 (LoRA fine-tuning):")
	print(f"  checkpoint_path={stage2_config['checkpoint_path']}")
	print(f"  head_init_checkpoint={stage2_config['head_init_checkpoint']}")
	print(f"  optimizer_name={stage2_config['optimizer_name']}")
	print(f"  scheduler={stage2_config['scheduler']}")
	print(f"  lora_targets={stage2_config['lora_targets']}")

	stage1_solution = get_solution("baseline_224_targeted_augmentations", stage1_config)
	stage1_solution.fit()
	stage1_checkpoint = Path(stage1_config["checkpoint_path"])
	if not stage1_checkpoint.exists():
		raise RuntimeError(f"Stage 1 checkpoint was not created: {stage1_checkpoint}")

	stage2_solution = get_solution("lora_uni_targeted_augmentations", stage2_config)
	stage2_solution.fit()
	stage2_solution.predict_test(output_csv=stage2_config["output_csv"])


if __name__ == "__main__":
	main()
