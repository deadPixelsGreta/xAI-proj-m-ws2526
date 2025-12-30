"""Shared argument parsing utilities for ensemble training and inference scripts."""

import argparse
from pathlib import Path


def add_common_args(parser: argparse.ArgumentParser):
    """Add common data and checkpoint path arguments."""
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        type=str,
        default="datasets",
        help="Path to dataset directory",
        dest="data_dir",
    )
    # Allow both --checkpoint-dir and --save-dir for compatibility
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        "--save-dir",
        "--save_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "checkpoints"),
        help="Directory for loading/saving model checkpoints",
        dest="checkpoint_dir",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing value (0.0 to 1.0)",
    )


def add_wandb_args(
    parser: argparse.ArgumentParser, default_project: str = "imagenet-subset"
):
    """Add Weights & Biases configuration arguments."""
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (overrides config/defaults)",
    )
    parser.add_argument(
        "--wandb-project",
        "--wandb_project",
        type=str,
        default=default_project,
        help="W&B project name",
        dest="wandb_project",
    )
    parser.add_argument(
        "--wandb-run-name",
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name",
        dest="wandb_run_name",
    )
    parser.add_argument(
        "--wandb-job-type",
        "--wandb_job_type",
        type=str,
        default=None,
        help="W&B job type",
        dest="wandb_job_type",
    )
    parser.add_argument(
        "--wandb-entity",
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity",
        dest="wandb_entity",
    )
    parser.add_argument(
        "--wandb-tags",
        "--wandb_tags",
        nargs="+",
        default=None,
        help="W&B tags",
        dest="wandb_tags",
    )
    parser.add_argument(
        "--wandb-notes",
        "--wandb_notes",
        type=str,
        default=None,
        help="W&B notes",
        dest="wandb_notes",
    )
    parser.add_argument(
        "--wandb-mode",
        "--wandb_mode",
        type=str,
        default=None,
        help="W&B mode (online/offline/disabled)",
        dest="wandb_mode",
    )
    parser.add_argument(
        "--wandb-dir",
        "--wandb_dir",
        type=str,
        default=None,
        help="W&B save directory",
        dest="wandb_dir",
    )
    parser.add_argument(
        "--wandb-group",
        "--wandb_group",
        type=str,
        default=None,
        help="W&B group name",
        dest="wandb_group",
    )


def add_model_args(parser: argparse.ArgumentParser, supported_models: list):
    """Add shared model architecture arguments."""
    parser.add_argument(
        "--model",
        type=str,
        default="densenet121",
        choices=supported_models,
        help="Model architecture backbone",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train/Load models from scratch without pretrained weights",
    )

    # Optimization arguments
    parser.add_argument(
        "--amp", action="store_true", help="Use Automatic Mixed Precision (AMP)"
    )
    parser.add_argument(
        "--grad-clip",
        "--grad_clip",
        type=float,
        default=0.0,
        help="Gradient clipping norm (0.0 to disable)",
        dest="grad_clip",
    )
    parser.add_argument(
        "--accum-steps",
        "--accum_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
        dest="accum_steps",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="step",
        choices=["step", "cosine", "onecycle"],
        help="Learning rate scheduler type",
    )
