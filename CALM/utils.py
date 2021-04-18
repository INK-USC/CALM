import glob
import os
import random
import re

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed relevant RNGs.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_val_loss(checkpoint_path: str) -> float:
    """Extract validation loss from checkpoint path.

    E.g. checkpoint path format: path_to_dir/checkpoint_epoch=4-val_loss=0.450662.ckpt

    Args:
        checkpoint_path: Path to checkpoint.

    Returns:
        Parsed validation loss, if available.
    """
    match = re.search("val_loss=(.+?).ckpt", checkpoint_path)
    if match:
        return float(match.group(1))
    else:
        raise ValueError


def extract_step_or_epoch(checkpoint_path: str) -> int:
    """Extract step or epoch number from checkpoint path.

    E.g. checkpoint path formats:
        - path_to_dir/checkpoint_epoch=4.ckpt
        - path_to_dir/checkpoint_epoch=4-step=50.ckpt

    Args:
        checkpoint_path: Path to checkpoint.

    Returns:
        Parsed step or epoch number, if available.
    """
    if "step" in checkpoint_path:
        regex = "step=(.+?).ckpt"
    else:
        regex = "epoch=(.+?).ckpt"

    match = re.search(regex, checkpoint_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError


def get_best_checkpoint(checkpoint_dir: str) -> str:
    """Get best checkpoint in directory.

    Args:
        checkpoint_dir: Directory of checkpoints.

    Returns:
        Path to best checkpoint.
    """
    checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.ckpt"))

    try:
        # Get the checkpoint with lowest validation loss
        sorted_list = sorted(checkpoint_list, key=lambda x: extract_val_loss(x.split("/")[-1]))
    except ValueError:
        # If validation loss is not present,
        # get the checkpoint with highest step number or epoch number
        sorted_list = sorted(
            checkpoint_list, key=lambda x: extract_step_or_epoch(x.split("/")[-1]), reverse=True
        )

    return sorted_list[0]
