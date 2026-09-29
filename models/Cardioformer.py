"""Cardioformer ECG backbone.

PROVENANCE / CLEAN-ROOM NOTE
---------------------------
This is an *independent clean-room re-implementation* of the Cardioformer encoder
written from the description in the paper (Mobin et al., "Cardioformer:
Advancing AI in ECG Analysis with Multi-Granularity Patching and ResNet",
arXiv:2505.05538, 2025). No code from the authors' repository was used, and the
implementation has not been verified against their released weights or results.

Consequences that must be stated in the thesis:
  * published Cardioformer numbers are NOT directly comparable to numbers from
    this code -- any ECG-only baseline here is "our re-implementation of
    Cardioformer", not "Cardioformer";
  * undocumented details (patch strides, router-token design, normalisation
    placement, initialisation) are our own choices and are documented in
    layers/Embed.py and layers/Cardioformer_EncDec.py;
  * a gap against the published results is therefore expected and should not be
    reported as a failure to reproduce the paper.

It follows the Time-Series-Library ``Model(configs)`` convention so it can be used
as a drop-in ECG encoder. The backbone is reusable both for classification and as
a feature extractor for the multimodal prognostic model
(``models.CardioformerCKD``).

Expected ``configs`` attributes
-------------------------------
enc_in           : number of ECG leads (channels), e.g. 12
seq_len          : number of timestamps in an ECG window
d_model          : token / model dimension (paper: 128)
n_heads          : attention heads
e_layers         : number of encoder layers (paper: 6)
d_ff             : feed-forward hidden dim (paper: 256)
dropout          : dropout rate
patch_len_list   : comma-separated granularities, e.g. "2,4,8,8,16,16,16,16,32"
num_class        : number of classes (classification task only)
cross_channel    : bool, use cross-channel patching (default True)
patch_encoder    : 'resnet' (default, as in the paper) | 'linear' (ablation)
inter_granularity: bool, stage-2 inter-granularity attention (default True;
                   False is an ablation)
"""

import torch
import torch.nn as nn

from layers.Embed import MultiGranularityEmbedding
from layers.Cardioformer_EncDec import CardioformerEncoder


def _parse_patch_lens(configs):
    raw = getattr(configs, "patch_len_list", "2,4,8,8,16,16,16,16,32")
    if isinstance(raw, (list, tuple)):
        return [int(p) for p in raw]
    return [int(p) for p in str(raw).split(",") if str(p).strip()]


class Model(nn.Module):
    """Cardioformer backbone. Set ``task_name='classification'`` for the original
    ECG-classification behaviour, or use ``encode()`` to obtain a pooled embedding
    for downstream multimodal fusion."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "classification")
        self.d_model = configs.d_model
        self.patch_lens = _parse_patch_lens(configs)
        self.n_gran = len(self.patch_lens)

        self.embedding = MultiGranularityEmbedding(
            n_leads=configs.enc_in,
            patch_lens=self.patch_lens,
            d_model=configs.d_model,
            dropout=configs.dropout,
            resnet_hidden=getattr(configs, "resnet_hidden", 64),
            resnet_blocks=getattr(configs, "resnet_blocks", 2),
            cross_channel=getattr(configs, "cross_channel", True),
            patch_encoder=getattr(configs, "patch_encoder", "resnet"),
        )
        self.encoder = CardioformerEncoder(
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            inter_granularity=getattr(configs, "inter_granularity", True),
        )
        self.embed_dim = configs.d_model  # per-granularity pooled dim

        if self.task_name == "classification":
            self.act = nn.GELU()
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.n_gran * configs.d_model, configs.num_class)

    # ---- core feature extractor --------------------------------------------
    def encode(self, x_enc, return_tokens=False):
        """x_enc: (B, seq_len, enc_in) in TSLib layout.

        Returns ``(B, G*d_model)``: the mean-pooled token of each granularity,
        concatenated. With ``return_tokens=True`` also returns the full patch
        token sequence ``(B, T, d_model)`` -- every token from every granularity
        concatenated along time, where ``T = sum_g ceil(seq_len / patch_len_g)``.
        The pooled vector alone throws that sequence away, which is why the
        cross-attention fusion needs this flag (see models/CardioformerCKD.py).
        """
        x = x_enc.permute(0, 2, 1).contiguous()   # -> (B, C, L)
        streams = self.embedding(x)               # list of (B, n_g, d_model)
        if return_tokens:
            enc, normed = self.encoder(streams, return_streams=True)
            tokens = torch.cat(normed, dim=1)     # (B, T, d_model)
            return enc.reshape(enc.size(0), -1), tokens
        enc = self.encoder(streams)               # (B, G, d_model)
        return enc.reshape(enc.size(0), -1)       # (B, G*d_model)

    def classification(self, x_enc, x_mark_enc):
        feats = self.encode(x_enc)
        feats = self.dropout(self.act(feats))
        return self.projection(feats)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        # default: return embedding
        return self.encode(x_enc)
