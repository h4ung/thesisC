"""1-D ResNet blocks.

In the original Cardioformer, instead of flattening a patch and applying a single
linear projection (as PatchTST / Medformer do), each raw patch is passed through a
small 1-D residual network that maps it directly to a token of dimension ``d_model``.
This module provides the residual building blocks used by the patch embedding.
"""

import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    """A pre-activation style 1-D residual block."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class ResNetPatchEncoder(nn.Module):
    """Map a raw patch of shape (B*, in_channels, patch_len) to a token (B*, d_model).

    Parameters
    ----------
    in_channels : int
        Number of input channels fed to the encoder. For *cross-channel* patching
        this equals the number of ECG leads (e.g. 12); for channel-independent
        patching this is 1.
    d_model : int
        Output token dimension.
    hidden : int
        Width of the residual stack.
    n_blocks : int
        Number of residual blocks.
    """

    def __init__(self, in_channels, d_model, hidden=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        blocks = []
        ch = hidden
        for i in range(n_blocks):
            out_ch = hidden * (2 ** i)
            blocks.append(BasicBlock1d(ch, out_ch, stride=1, dropout=dropout))
            ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(ch, d_model)

    def forward(self, x):
        # x: (N, in_channels, patch_len)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)  # (N, ch)
        x = self.proj(x)              # (N, d_model)
        return x
