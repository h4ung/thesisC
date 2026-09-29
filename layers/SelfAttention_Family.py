"""Self-attention building blocks (scaled dot-product, multi-head).

Kept deliberately small and dependency-free so the encoder is easy to read and
swap with the upstream Time-Series-Library attention if desired.
"""

import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q,k,v: (B, H, T, d_k)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)

    def _shape(self, x, B):
        # (B, T, d_model) -> (B, H, T, d_k)
        return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, attn_mask=None):
        B = query.size(0)
        q = self._shape(self.q_proj(query), B)
        k = self._shape(self.k_proj(key), B)
        v = self._shape(self.v_proj(value), B)
        out, attn = self.attn(q, k, v, attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.out_proj(out), attn
