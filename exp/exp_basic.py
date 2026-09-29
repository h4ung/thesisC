"""Base experiment class (Time-Series-Library style)."""

import os

import torch


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            dev = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            dev = torch.device("cpu")
            print("Use CPU")
        return dev

    # ---- checkpoints -----------------------------------------------------
    def checkpoint_dir(self):
        return os.path.join(self.args.checkpoints, self.args.setting)

    def checkpoint_file(self):
        """Explicit --checkpoint_path if given, else <checkpoints>/<setting>/checkpoint.pth."""
        explicit = getattr(self.args, "checkpoint_path", None)
        if explicit:
            return explicit
        return os.path.join(self.checkpoint_dir(), "checkpoint.pth")

    def load_checkpoint(self, path=None, required=True):
        """Load model weights from disk.

        ``test()`` must call this: previously only ``train()`` restored the best
        weights at the end of the loop, so ``--is_training 0`` silently evaluated
        a randomly initialised model.
        """
        path = path or self.checkpoint_file()
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(
                    f"No checkpoint at {path}. Train first (--is_training 1), or point "
                    f"--checkpoint_path at an existing .pth."
                )
            print(f"[checkpoint] none found at {path}; using current weights")
            return False
        try:  # torch >= 2.6 defaults to weights_only=True; be explicit either way
            state = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # pragma: no cover - older torch
            state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        print(f"[checkpoint] loaded {path}")
        return True

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
