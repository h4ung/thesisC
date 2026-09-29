"""CardioformerCKD: multimodal prognostic model for incident CKD.

Architecture
------------
        ECG window (B, L, 12) ──► Cardioformer backbone ──► z_ecg (B, G*d_model)
                                                               │
   EHR features (static / seq) ──► EHR encoder ──► z_ehr (B, d_ehr)
                                                               │
                          ┌────────── fusion (concat | gated | cross-attn) ──────────┐
                          ▼                                                            ▼
                   fused (B, d_fuse)                               prognostic RISK layer (Thesis C)
                                              discrete-time survival (default) | Cox PH | [binary: ablation only]
                                                        -> risk score + absolute risk CIF(t) per patient
                                                        -> risk groups (utils/risk_stratification.py)

The model is *prognostic*: features come from a window strictly before the index
time; labels describe CKD onset within a future horizon. Leakage control lives in
the cohort builder (``data_preprocessing/build_cohort.py``), not here.

``configs`` reuses the Cardioformer attributes (see models/Cardioformer.py) plus:
ehr_in_dim, ehr_mode ('mlp'|'seq'|'identity'), fusion ('concat'|'gated'|'cross'),
cross_attn_tokens ('patches'|'granularity', cross fusion only),
head ('survival'|'cox'|'binary'), head_hidden (0 = linear head),
n_intervals (survival only), use_ecg, use_ehr.

Provenance: the ECG backbone is a clean-room re-implementation of Cardioformer
from the paper (Mobin et al., 2025, arXiv:2505.05538), not the authors' code.
See the note in models/Cardioformer.py.
"""

import torch
import torch.nn as nn

from models.Cardioformer import Model as CardioformerBackbone
from models.ehr_encoder import EHRIdentityEncoder, EHRMLPEncoder, EHRSequenceEncoder
from models.heads import BinaryHorizonHead, CoxRiskHead, DiscreteTimeSurvivalHead


class GatedFusion(nn.Module):
    """Modality gating: learn per-dimension weights to combine the two embeddings."""

    def __init__(self, d_ecg, d_ehr, d_out):
        super().__init__()
        self.proj_ecg = nn.Linear(d_ecg, d_out)
        self.proj_ehr = nn.Linear(d_ehr, d_out)
        self.gate = nn.Linear(d_out * 2, d_out)

    def forward(self, z_ecg, z_ehr):
        a = self.proj_ecg(z_ecg)
        b = self.proj_ehr(z_ehr)
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return g * a + (1 - g) * b


class CrossAttentionFusion(nn.Module):
    """EHR embedding attends over the ECG token sequence.

    ``ecg_tokens`` is the *full* multi-granularity patch sequence (B, T, d_model)
    with T = sum_g ceil(seq_len / patch_len_g) -- a few hundred tokens for a 250-
    sample window. Attending over these lets the EHR query select which parts of
    the waveform (and which scale) matter for this patient.

    Set ``configs.cross_attn_tokens='granularity'`` to attend over only the G
    pooled per-granularity summaries instead (the original behaviour, kept for
    ablation; it is a much weaker form of cross-attention because the pooling has
    already discarded the within-granularity structure).
    """

    def __init__(self, d_ecg_token, d_ehr, d_out, n_heads=4, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(d_ehr, d_out)
        self.kv = nn.Linear(d_ecg_token, d_out)
        self.attn = nn.MultiheadAttention(d_out, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_out)
        self.out = nn.Linear(d_out + d_ehr, d_out)

    def forward(self, ecg_tokens, z_ehr, key_padding_mask=None):
        # ecg_tokens: (B, T, d_ecg_token); z_ehr: (B, d_ehr)
        q = self.q(z_ehr).unsqueeze(1)                       # (B, 1, d_out)
        kv = self.kv(ecg_tokens)                             # (B, T, d_out)
        ctx, attn = self.attn(q, kv, kv, key_padding_mask=key_padding_mask,
                              need_weights=True)             # ctx (B,1,d_out), attn (B,1,T)
        self.last_attn = attn.detach()                       # kept for interpretability plots
        ctx = self.norm(q + ctx).squeeze(1)                  # residual around the query
        return self.out(torch.cat([ctx, z_ehr], dim=-1))


def _pick_heads(d_out, desired):
    """Largest head count <= desired that divides d_out (MultiheadAttention requires it)."""
    for h in range(max(1, min(int(desired), int(d_out))), 0, -1):
        if d_out % h == 0:
            return h
    return 1


class CardioformerCKD(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.use_ecg = getattr(configs, "use_ecg", True)
        self.use_ehr = getattr(configs, "use_ehr", True)
        self.fusion_mode = getattr(configs, "fusion", "gated")
        # 'patches' -> attend over every patch token; 'granularity' -> only the G
        # pooled granularity summaries (legacy behaviour, weaker).
        self.cross_attn_tokens = getattr(configs, "cross_attn_tokens", "patches")
        self.head_type = getattr(configs, "head", "survival")
        self.d_model = configs.d_model

        d_fuse = getattr(configs, "d_fuse", 256)

        # --- ECG branch -----------------------------------------------------
        if self.use_ecg:
            backbone_cfg = configs
            setattr(backbone_cfg, "task_name", "representation")
            self.ecg = CardioformerBackbone(backbone_cfg)
            self.n_gran = self.ecg.n_gran
            d_ecg = self.n_gran * configs.d_model
        else:
            self.ecg = None
            d_ecg = 0

        # --- EHR branch -----------------------------------------------------
        if self.use_ehr:
            ehr_mode = getattr(configs, "ehr_mode", "mlp")
            if ehr_mode == "seq":
                self.ehr = EHRSequenceEncoder(
                    n_features=configs.ehr_in_dim, d_model=configs.d_model,
                    dropout=getattr(configs, "ehr_dropout", 0.2),
                )
            elif ehr_mode == "identity":
                self.ehr = EHRIdentityEncoder(configs.ehr_in_dim)
            else:
                self.ehr = EHRMLPEncoder(
                    in_dim=configs.ehr_in_dim, embed_dim=configs.d_model,
                    dropout=getattr(configs, "ehr_dropout", 0.2),
                )
            d_ehr = self.ehr.embed_dim
        else:
            self.ehr = None
            d_ehr = 0

        # --- fusion ---------------------------------------------------------
        if self.use_ecg and self.use_ehr:
            if self.fusion_mode == "concat":
                self.fuse = None
                d_fuse = d_ecg + d_ehr
            elif self.fusion_mode == "gated":
                self.fuse = GatedFusion(d_ecg, d_ehr, d_fuse)
            elif self.fusion_mode == "cross":
                self.fuse = CrossAttentionFusion(
                    configs.d_model, d_ehr, d_fuse,
                    n_heads=_pick_heads(d_fuse, getattr(configs, "n_heads", 4)),
                    dropout=getattr(configs, "dropout", 0.1),
                )
            else:
                raise ValueError(f"unknown fusion: {self.fusion_mode}")
        else:
            self.fuse = None
            d_fuse = d_ecg + d_ehr

        # --- prognostic risk layer -----------------------------------------
        head_hidden = int(getattr(configs, "head_hidden", 128))
        head_dropout = float(getattr(configs, "head_dropout", 0.2))
        if self.head_type == "survival":
            self.head = DiscreteTimeSurvivalHead(
                d_fuse, n_intervals=getattr(configs, "n_intervals", 8),
                hidden=head_hidden, dropout=head_dropout,
            )
        elif self.head_type == "cox":
            self.head = CoxRiskHead(d_fuse, hidden=head_hidden, dropout=head_dropout)
        elif self.head_type == "binary":
            self.head = BinaryHorizonHead(d_fuse, hidden=head_hidden, dropout=head_dropout)
        else:
            raise ValueError(f"unknown head: {self.head_type}")

    def _encode_ecg(self, x_ecg, need_tokens=False):
        """Return the pooled ECG embedding and, when needed, the token sequence.

        ``need_tokens`` asks the backbone for the full multi-granularity patch
        sequence (B, T, d_model). The previous version reshaped the pooled vector
        into (B, G, d_model), so 'cross' fusion could only ever attend over G
        granularity summaries -- not over the waveform itself.
        """
        if not need_tokens:
            return self.ecg.encode(x_ecg), None
        if self.cross_attn_tokens == "granularity":
            feats = self.ecg.encode(x_ecg)                              # (B, G*d_model)
            return feats, feats.view(feats.size(0), self.n_gran, self.d_model)
        feats, tokens = self.ecg.encode(x_ecg, return_tokens=True)      # (B, T, d_model)
        return feats, tokens

    def forward(self, x_ecg=None, ehr=None, ehr_mask=None):
        z_ecg = z_ehr = None
        ecg_tokens = None

        need_tokens = self.use_ecg and self.use_ehr and self.fusion_mode == "cross"
        if self.use_ecg:
            z_ecg, ecg_tokens = self._encode_ecg(x_ecg, need_tokens=need_tokens)
        if self.use_ehr:
            if isinstance(self.ehr, EHRSequenceEncoder):
                z_ehr = self.ehr(ehr, key_padding_mask=ehr_mask)
            else:
                z_ehr = self.ehr(ehr)

        if self.use_ecg and self.use_ehr:
            if self.fusion_mode == "concat":
                fused = torch.cat([z_ecg, z_ehr], dim=-1)
            elif self.fusion_mode == "gated":
                fused = self.fuse(z_ecg, z_ehr)
            else:  # cross
                fused = self.fuse(ecg_tokens, z_ehr)
        elif self.use_ecg:
            fused = z_ecg
        else:
            fused = z_ehr

        return self.head(fused)

    # convenience for survival inference
    def predict_cif(self, x_ecg=None, ehr=None, ehr_mask=None):
        assert self.head_type == "survival"
        logits = self.forward(x_ecg, ehr, ehr_mask)
        surv, cif = DiscreteTimeSurvivalHead.survival_from_hazards(logits)
        return surv, cif
