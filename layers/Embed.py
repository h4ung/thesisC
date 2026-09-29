"""Multi-granularity, cross-channel patch embedding for ECG signals.

This re-implements the token-embedding idea described in the Cardioformer paper
(Mobin et al., 2025, arXiv:2505.05538) using the Time-Series-Library interface:

* The signal is patched at *several granularities* (a list of patch lengths).
* Patches are *cross-channel*: a patch keeps all leads together so inter-lead
  correlations can be modelled (rather than treating each lead independently).
* Each raw patch is mapped to a ``d_model`` token by a small 1-D ResNet
  (``ResNetPatchEncoder``) instead of a flatten + linear projection.

Output is a list (one entry per granularity) of token tensors with shape
``(B, num_patches_g, d_model)`` plus a positional embedding.
"""

import torch
import torch.nn as nn

from layers.ResNet import ResNetPatchEncoder


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class LinearPatchEncoder(nn.Module):
    """Flatten a (N, C, patch_len) patch and project it linearly to d_model.

    Used only for the ablation that replaces Cardioformer's ResNet patch encoder
    with the flatten-and-project embedding of PatchTST / Medformer.
    """

    def __init__(self, in_channels, patch_len, d_model):
        super().__init__()
        self.proj = nn.Linear(in_channels * patch_len, d_model)

    def forward(self, x):
        return self.proj(x.reshape(x.size(0), -1))


class CrossChannelPatch(nn.Module):
    """Patch a multi-lead signal at a single granularity (patch length).

    Input : (B, C, L)  -- C leads, L timestamps
    Output: (B, num_patches, d_model)
    """

    def __init__(self, n_leads, patch_len, d_model, stride=None,
                 resnet_hidden=64, resnet_blocks=2, dropout=0.1, cross_channel=True,
                 patch_encoder="resnet"):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride or patch_len  # non-overlapping by default
        self.cross_channel = cross_channel
        in_channels = n_leads if cross_channel else 1
        if patch_encoder == "resnet":
            self.encoder = ResNetPatchEncoder(
                in_channels=in_channels, d_model=d_model,
                hidden=resnet_hidden, n_blocks=resnet_blocks, dropout=dropout,
            )
        elif patch_encoder == "linear":
            # Ablation: PatchTST/Medformer-style flatten + linear projection
            self.encoder = LinearPatchEncoder(in_channels, patch_len, d_model)
        else:
            raise ValueError(f"unknown patch_encoder {patch_encoder!r}")
        self.n_leads = n_leads

    def forward(self, x):
        B, C, L = x.shape
        # pad so L is divisible by stride
        if L % self.stride != 0:
            pad = self.stride - (L % self.stride)
            x = nn.functional.pad(x, (0, pad))
            L = x.shape[-1]
        # unfold into patches along time: (B, C, num_patches, patch_len)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        num_patches = patches.shape[2]

        if self.cross_channel:
            # (B, num_patches, C, patch_len) -> (B*num_patches, C, patch_len)
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(B * num_patches, C, self.patch_len)
            tokens = self.encoder(patches)                  # (B*num_patches, d_model)
            tokens = tokens.view(B, num_patches, -1)
        else:
            # channel independent: (B, C, num_patches, patch_len)
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(B * num_patches * C, 1, self.patch_len)
            tokens = self.encoder(patches)
            tokens = tokens.view(B, num_patches, C, -1).mean(dim=2)
        return tokens, num_patches


class MultiGranularityEmbedding(nn.Module):
    """Build one token sequence per granularity from a raw ECG window.

    Parameters
    ----------
    n_leads : int
        Number of ECG leads (channels).
    patch_lens : list[int]
        List of patch lengths. Repetition is allowed, e.g.
        ``[2, 4, 8, 8, 16, 16, 16, 16, 32, ...]`` as in the paper.
    d_model : int
        Token dimension.
    """

    def __init__(self, n_leads, patch_lens, d_model, dropout=0.1,
                 resnet_hidden=64, resnet_blocks=2, cross_channel=True,
                 patch_encoder="resnet"):
        super().__init__()
        self.patch_lens = list(patch_lens)
        self.embeds = nn.ModuleList([
            CrossChannelPatch(
                n_leads=n_leads, patch_len=p, d_model=d_model,
                resnet_hidden=resnet_hidden, resnet_blocks=resnet_blocks,
                dropout=dropout, cross_channel=cross_channel,
                patch_encoder=patch_encoder,
            )
            for p in self.patch_lens
        ])
        self.pos = PositionalEmbedding(d_model)
        # learnable granularity (scale) embedding added to every token of a granularity
        self.gran_embed = nn.Parameter(torch.randn(len(self.patch_lens), 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, C, L). Returns list of (B, n_patches_g, d_model)."""
        out = []
        for g, embed in enumerate(self.embeds):
            tokens, _ = embed(x)                       # (B, n_g, d_model)
            tokens = tokens + self.pos(tokens) + self.gran_embed[g]
            out.append(self.dropout(tokens))
        return out
