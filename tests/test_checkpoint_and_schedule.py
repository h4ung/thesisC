"""Guards for two silent-failure bugs.

1. ``test()`` used to evaluate whatever weights were in memory. With
   ``--is_training 0`` that is a randomly initialised model, and the reported
   test metrics are meaningless. ``Exp_Basic.load_checkpoint`` now does the load
   and raises if the checkpoint is missing.
2. ``lradj='type1'`` halves the LR every epoch (1e-4 -> ~1e-7 by epoch 10), so
   training stops learning almost immediately. The default is now cosine.
"""

import argparse

import pytest
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate


class TinyModel(nn.Module):
    def __init__(self, bias=0.0):
        super().__init__()
        self.lin = nn.Linear(4, 2)
        with torch.no_grad():
            self.lin.bias.fill_(bias)

    def forward(self, x):
        return self.lin(x)


class TinyExp(Exp_Basic):
    def _build_model(self):
        return TinyModel()


def _args(tmp_path, **kw):
    d = dict(use_gpu=False, gpu=0, checkpoints=str(tmp_path), setting="unit",
             checkpoint_path=None, learning_rate=1e-4, train_epochs=10,
             lradj="cosine", min_lr=1e-6, lr_decay_every=5, lr_decay_rate=0.5)
    d.update(kw)
    return argparse.Namespace(**d)


# ---- checkpoint loading -------------------------------------------------
def test_load_checkpoint_restores_saved_weights(tmp_path):
    exp = TinyExp(_args(tmp_path))
    trained = TinyModel(bias=3.14)
    EarlyStopping()(0.9, trained, exp.checkpoint_dir())   # writes checkpoint.pth

    assert not torch.allclose(exp.model.lin.bias, trained.lin.bias)
    exp.load_checkpoint()
    assert torch.allclose(exp.model.lin.bias, trained.lin.bias)


def test_missing_checkpoint_raises_instead_of_scoring_random_weights(tmp_path):
    exp = TinyExp(_args(tmp_path, setting="never_trained"))
    with pytest.raises(FileNotFoundError):
        exp.load_checkpoint()


def test_explicit_checkpoint_path_wins(tmp_path):
    exp = TinyExp(_args(tmp_path))
    other = tmp_path / "elsewhere.pth"
    torch.save(TinyModel(bias=-1.5).state_dict(), other)
    exp.args.checkpoint_path = str(other)
    assert exp.checkpoint_file() == str(other)
    exp.load_checkpoint()
    assert torch.allclose(exp.model.lin.bias, torch.full((2,), -1.5))


# ---- LR schedule --------------------------------------------------------
def _lr_after(args, epoch):
    opt = torch.optim.Adam(TinyModel().parameters(), lr=args.learning_rate)
    adjust_learning_rate(opt, epoch, args)
    return opt.param_groups[0]["lr"]


def test_cosine_decays_gently_and_is_the_default(tmp_path):
    args = _args(tmp_path, lradj="cosine", train_epochs=20)
    assert _lr_after(args, 1) == pytest.approx(1e-4)          # epoch 1: full LR
    mid = _lr_after(args, 11)
    assert 0.3e-4 < mid < 0.7e-4                              # ~half-way by mid-run
    assert _lr_after(args, 10) > 1e-5                         # not collapsed by epoch 10


def test_type1_collapse_is_reproduced_but_floored(tmp_path):
    args = _args(tmp_path, lradj="type1")
    assert _lr_after(args, 10) == pytest.approx(max(1e-4 * 0.5 ** 9, args.min_lr))
    assert _lr_after(args, 10) < 1e-6 * 1.0001                # this is why it is not the default


def test_step_schedule(tmp_path):
    args = _args(tmp_path, lradj="step", lr_decay_every=5, lr_decay_rate=0.5)
    assert _lr_after(args, 1) == pytest.approx(1e-4)
    assert _lr_after(args, 5) == pytest.approx(1e-4)
    assert _lr_after(args, 6) == pytest.approx(0.5e-4)


def test_none_schedule_is_constant(tmp_path):
    args = _args(tmp_path, lradj="none")
    assert _lr_after(args, 50) == pytest.approx(1e-4)
