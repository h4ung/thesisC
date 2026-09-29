"""EHR encoders for the structured MIMIC-IV branch.

Two options are provided:

* ``EHRMLPEncoder`` — for a single static feature vector per patient (aggregated
  labs + demographics + comorbidity flags). Robust default.
* ``EHRSequenceEncoder`` — a lightweight Transformer over a *time series* of
  longitudinal measurements (e.g. monthly-binned labs before the index time),
  for users who build a temporal EHR tensor.

* ``EHRIdentityEncoder`` — passes the standardised features through unchanged.
  Combined with a linear Cox head this is a classical Cox PH model, used as the
  non-deep baseline in the ablation study.

All expose ``.embed_dim`` and return a (B, embed_dim) embedding.
"""

import torch
import torch.nn as nn


class EHRMLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden=256, embed_dim=128, depth=3, dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(depth):
            out = hidden if i < depth - 1 else embed_dim
            layers += [
                nn.Linear(d, out),
                nn.LayerNorm(out),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            d = out
        self.net = nn.Sequential(*layers)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B, in_dim)
        return self.net(x)


class EHRIdentityEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.embed_dim = in_dim

    def forward(self, x):
        return x


class EHRSequenceEncoder(nn.Module):
    """Transformer encoder over (B, T, F) longitudinal EHR features.

    A learnable [CLS] token summarises the sequence. A padding mask (B, T) with
    True for valid steps may be supplied.
    """

    def __init__(self, n_features, d_model=128, n_heads=4, n_layers=2,
                 d_ff=256, dropout=0.2, max_len=64):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.embed_dim = d_model

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, F); key_padding_mask: (B, T) True == PAD
        B, T, _ = x.shape
        h = self.input_proj(x)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos[:, : T + 1]
        if key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.norm(h[:, 0])  # CLS embedding
