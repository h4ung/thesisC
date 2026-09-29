"""ECG augmentations (applied to training windows only).

Mirrors the augmentation menu described in the Cardioformer paper: randomly pick
from {none, jitter, scale, mask}. Operates on tensors of shape (C, L).
"""

import numpy as np
import torch


def jitter(x, sigma=0.03):
    return x + torch.randn_like(x) * sigma


def scaling(x, sigma=0.1):
    factor = 1.0 + torch.randn(x.size(0), 1, device=x.device) * sigma
    return x * factor


def masking(x, mask_ratio=0.1):
    L = x.size(-1)
    n = int(L * mask_ratio)
    if n <= 0:
        return x
    start = np.random.randint(0, max(1, L - n))
    x = x.clone()
    x[:, start : start + n] = 0.0
    return x


_AUGS = {"none": lambda x: x, "jitter": jitter, "scale": scaling, "mask": masking}


def random_augment(x, options=("none", "jitter", "scale", "mask")):
    """x: (C, L) tensor. Apply one randomly chosen augmentation."""
    name = np.random.choice(list(options))
    return _AUGS[name](x)
