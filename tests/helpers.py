"""Shared test helpers."""

import numpy as np
import torch


def ss_to_numpy(ss):
    """Convert sign sequence to numpy for comparison (handles torch.Tensor or ndarray)."""
    if isinstance(ss, torch.Tensor):
        return ss.detach().cpu().numpy()
    return np.asarray(ss)
