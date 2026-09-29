"""Misc training utilities (kept compatible with Time-Series-Library)."""

import os
import random

import numpy as np
import torch


def set_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adjust_learning_rate(optimizer, epoch, args):
    """Set the LR for ``epoch`` (1-indexed) according to ``args.lradj``.

    Schedules
    ---------
    cosine : lr * 0.5 * (1 + cos(pi * (epoch-1) / train_epochs))   [default]
    step   : lr * lr_decay_rate ** ((epoch-1) // lr_decay_every)
    type1  : lr * 0.5 ** (epoch-1)  -- legacy Time-Series-Library schedule. This
             halves the LR *every* epoch, so by epoch 10 it is ~1e-3 of the
             initial value and training has effectively stopped. Kept only so
             older runs remain reproducible; do not use it for new experiments.
    none   : constant LR.

    Every schedule is floored at ``args.min_lr`` (default 1e-6).
    """
    base = args.learning_rate
    mode = getattr(args, "lradj", "cosine")
    min_lr = float(getattr(args, "min_lr", 1e-6))
    total = max(int(getattr(args, "train_epochs", 1)), 1)

    if mode in ("none", "const", "constant"):
        return None
    if mode == "cosine":
        lr = base * 0.5 * (1 + np.cos(np.pi * min((epoch - 1) / total, 1.0)))
    elif mode == "step":
        every = max(int(getattr(args, "lr_decay_every", 5)), 1)
        rate = float(getattr(args, "lr_decay_rate", 0.5))
        lr = base * (rate ** ((epoch - 1) // every))
    elif mode == "type1":
        lr = base * (0.5 ** (epoch - 1))
    else:
        return None

    lr = max(lr, min_lr)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


class EarlyStopping:
    """Stop when the monitored score stops improving. Higher == better."""

    def __init__(self, patience=3, verbose=True, delta=0.0, mode="max"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, score, model, path):
        improved = (
            self.best is None
            or (self.mode == "max" and score > self.best + self.delta)
            or (self.mode == "min" and score < self.best - self.delta)
        )
        if improved:
            self.best = score
            self._save(model, path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, model, path):
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        if self.verbose:
            print(f"  -> validation improved ({self.best:.5f}); checkpoint saved")
