"""Prognostic risk-prediction heads (the "prognostic layer").

Thesis C change: the prognostic layer is a **risk-prediction** layer, not a
binary classifier. Every head below is wrapped so that, at inference, the model
produces the same two things for every patient:

* ``risk_score``   a continuous ranking score (higher = CKD expected sooner), used
                   for the C-index and for risk stratification;
* ``abs_risk(t)``  an absolute, calibrated probability of incident CKD by time t
                   (the cumulative incidence function, CIF), used for
                   calibration, the Brier score and absolute-risk cut-points.

Heads
-----
``DiscreteTimeSurvivalHead`` (default, ``--head survival``)
    K conditional hazards h_k = P(event in interval k | event-free at start of k).
    S(t_k) = prod_{j<=k} (1 - h_j), CIF = 1 - S. Handles right-censoring and gives
    a full risk curve over the horizon (Gensheimer & Narasimhan, 2019).

``CoxRiskHead`` (``--head cox``)
    A single log-risk eta = f(x) trained with the Cox partial likelihood
    (DeepSurv; Katzman et al., 2018). Absolute risk comes from a Breslow
    baseline hazard fitted on the training set after training:
    CIF(t | x) = 1 - exp(-H0(t) * exp(eta)).

``BinaryHorizonHead`` (``--head binary``)
    Single logit for "CKD by the horizon". Kept **only as an ablation baseline**:
    it discards patients censored before the horizon and gives no risk curve.

``head_hidden=0`` makes any head a single linear layer, which with
``--ehr_mode identity --use_ecg 0 --head cox`` reproduces a classical linear Cox
proportional-hazards model on the EHR features (the non-deep baseline).
"""

import numpy as np
import torch
import torch.nn as nn


def _mlp(in_dim, out_dim, hidden, dropout):
    if not hidden:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class BinaryHorizonHead(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = _mlp(in_dim, 1, hidden, dropout)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,) logit


class DiscreteTimeSurvivalHead(nn.Module):
    """Outputs K hazard logits; h_k = P(event in interval k | survived to k)."""

    def __init__(self, in_dim, n_intervals, hidden=128, dropout=0.2):
        super().__init__()
        self.n_intervals = n_intervals
        self.net = _mlp(in_dim, n_intervals, hidden, dropout)

    def forward(self, x):
        return self.net(x)  # (B, K) hazard logits

    @staticmethod
    def survival_from_hazards(hazard_logits):
        """Return survival S(t_k) = prod_{j<=k} (1 - h_j) and CIF = 1 - S."""
        h = torch.sigmoid(hazard_logits)               # (B, K)
        surv = torch.cumprod(1.0 - h, dim=1)           # (B, K)
        cif = 1.0 - surv
        return surv, cif


class CoxRiskHead(nn.Module):
    """Outputs a scalar log-risk eta per patient (proportional hazards)."""

    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = _mlp(in_dim, 1, hidden, dropout)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,) log-risk


# --------------------------------------------------------------------------
# Breslow baseline hazard for the Cox head
# --------------------------------------------------------------------------
class BreslowBaseline:
    """Breslow estimator of the baseline cumulative hazard H0(t).

    H0(t) = sum_{t_i <= t, event} d_i / sum_{j: T_j >= t_i} exp(eta_j)

    Fitted on the (horizon-censored) training set with the trained network's
    log-risks, then used to turn a log-risk into an absolute risk.
    """

    def __init__(self):
        self.times_ = None
        self.cumhaz_ = None

    def fit(self, log_risk, times, events):
        eta = np.asarray(log_risk, dtype=np.float64).ravel()
        t = np.asarray(times, dtype=np.float64).ravel()
        e = np.asarray(events).ravel().astype(bool)
        eta = eta - eta.max()                       # numerical stability (cancels in ratio)
        self._shift = float(np.asarray(log_risk).max()) if eta.size else 0.0
        w = np.exp(eta)
        order = np.argsort(-t, kind="stable")
        t_s, w_s = t[order], w[order]
        cum_w = np.cumsum(w_s)                      # risk-set sums, descending time
        uniq = np.unique(t[e])
        haz = []
        for u in uniq:
            d = int(((t == u) & e).sum())
            # risk set = T >= u -> last index in descending order with t_s >= u
            k = np.searchsorted(-t_s, -u, side="right") - 1
            haz.append(d / cum_w[k])
        self.times_ = uniq
        self.cumhaz_ = np.cumsum(haz) if haz else np.array([])
        return self

    def cumulative_hazard(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if self.times_ is None or self.times_.size == 0:
            return np.zeros_like(t)
        idx = np.searchsorted(self.times_, t, side="right") - 1
        out = np.zeros_like(t)
        out[idx >= 0] = self.cumhaz_[idx[idx >= 0]]
        return out

    def abs_risk(self, log_risk, t):
        """CIF(t | x) = 1 - exp(-H0(t) * exp(eta)). Returns (N, len(t))."""
        eta = np.asarray(log_risk, dtype=np.float64).ravel() - self._shift
        H0 = self.cumulative_hazard(t)                          # (T,)
        return 1.0 - np.exp(-np.outer(np.exp(eta), H0))

    def state_dict(self):
        return {"times": self.times_.tolist(), "cumhaz": self.cumhaz_.tolist(),
                "shift": self._shift}

    def load_state_dict(self, d):
        self.times_ = np.asarray(d["times"], dtype=np.float64)
        self.cumhaz_ = np.asarray(d["cumhaz"], dtype=np.float64)
        self._shift = float(d.get("shift", 0.0))
        return self


def horizon_bin_index(horizon_days, cohort_horizon_days, n_intervals):
    """Index of the survival bin whose right edge is the first >= ``horizon_days``.

    Bins are equal-width over [0, cohort_horizon_days] (see build_cohort.py), so
    CIF[:, k] is the risk by (k + 1) * cohort_horizon_days / n_intervals.
    """
    if horizon_days > cohort_horizon_days + 1e-9:
        raise ValueError(f"eval horizon {horizon_days}d exceeds the cohort horizon "
                         f"{cohort_horizon_days}d; the survival head cannot extrapolate")
    k = int(np.ceil(horizon_days / cohort_horizon_days * n_intervals - 1e-9)) - 1
    return int(min(max(k, 0), n_intervals - 1))
