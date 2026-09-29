"""Cardioformer encoder: two-stage multi-granularity self-attention.

Each encoder layer performs:
  1. **Intra-granularity** attention — tokens attend within their own granularity.
  2. **Inter-granularity** attention — a small number of summary/router tokens
     (one per granularity) exchange information across granularities, then the
     fused context is broadcast back into each granularity stream.

This mirrors the "two-stage multi-granularity self-attention (intra- and
inter-granularity)" described in the Cardioformer paper while staying within the
Time-Series-Library coding style.
"""

import torch
import torch.nn as nn

from layers.SelfAttention_Family import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TwoStageEncoderLayer(nn.Module):
    """``inter_granularity=False`` skips stage 2 (ablation): each granularity is
    then encoded independently and never exchanges information with the others.
    The inter-granularity parameters are still created so state dicts keep the
    same keys, but they receive no gradient."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, inter_granularity=True):
        super().__init__()
        self.inter_granularity = inter_granularity
        # intra-granularity
        self.intra_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.intra_norm = nn.LayerNorm(d_model)
        # inter-granularity (across router tokens)
        self.inter_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.inter_norm = nn.LayerNorm(d_model)
        # position-wise FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, streams):
        """streams: list of (B, n_g, d_model), one per granularity."""
        # ---- Stage 1: intra-granularity self-attention -------------------
        updated = []
        routers = []
        for s in streams:
            a, _ = self.intra_attn(s, s, s)
            s = self.intra_norm(s + self.dropout(a))
            updated.append(s)
            routers.append(s.mean(dim=1, keepdim=True))  # (B,1,d) summary token

        # ---- Stage 2: inter-granularity attention over router tokens -----
        if not self.inter_granularity:
            return [self.ffn_norm(s + self.dropout(self.ffn(s))) for s in updated]
        router_seq = torch.cat(routers, dim=1)            # (B, G, d_model)
        r, _ = self.inter_attn(router_seq, router_seq, router_seq)
        router_seq = self.inter_norm(router_seq + self.dropout(r))  # (B, G, d_model)

        # broadcast fused context back into each granularity stream + FFN
        out = []
        for g, s in enumerate(updated):
            ctx = router_seq[:, g : g + 1, :]             # (B,1,d_model)
            s = s + ctx                                   # inject cross-granularity context
            s = self.ffn_norm(s + self.dropout(self.ffn(s)))
            out.append(s)
        return out


class CardioformerEncoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1, inter_granularity=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoStageEncoderLayer(d_model, n_heads, d_ff, dropout,
                                 inter_granularity=inter_granularity)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, streams, return_streams=False):
        """streams: list of (B, n_g, d_model).

        Returns the per-granularity pooled representation ``(B, G, d_model)``.
        With ``return_streams=True`` the *full* normalised token sequences are
        also returned, so a downstream module (e.g. the cross-attention fusion in
        ``models.CardioformerCKD``) can attend over every patch token instead of
        only the G pooled summaries.
        """
        for layer in self.layers:
            streams = layer(streams)
        normed = [self.norm(s) for s in streams]               # list of (B, n_g, d_model)
        pooled = torch.stack([s.mean(dim=1) for s in normed], dim=1)   # (B, G, d_model)
        if return_streams:
            return pooled, normed
        return pooled
