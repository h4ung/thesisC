"""Loss functions for prognostic CKD training.

* ``focal_bce`` — class-imbalance-aware loss for the fixed-horizon binary head.
* ``discrete_time_nll`` — negative log-likelihood for the discrete-time survival
  head, correctly handling right-censoring.
* ``cox_ph_loss`` — Cox partial likelihood (Breslow ties) for the Cox risk head.
"""

import torch
import torch.nn.functional as F


def focal_bce(logits, targets, alpha=0.25, gamma=2.0, pos_weight=None):
    """Binary focal loss. ``logits`` and ``targets`` are (B,)."""
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
        pos_weight=pos_weight if pos_weight is None else torch.as_tensor(pos_weight, device=logits.device),
    )
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * bce
    return loss.mean()


def discrete_time_nll(hazard_logits, event_bin, event_indicator):
    """Negative log-likelihood for a discrete-time survival model.

    Parameters
    ----------
    hazard_logits : (B, K) logits of conditional hazards h_k.
    event_bin     : (B,) integer index in [0, K-1]. For an event, the interval the
                    event occurred in; for a censored subject, the last interval
                    fully observed.
    event_indicator : (B,) 1 if the event (CKD onset) was observed, 0 if censored.

    Likelihood for an event at bin e:  h_e * prod_{j<e} (1 - h_j)
    Likelihood for censoring at bin c:        prod_{j<=c} (1 - h_j)
    """
    B, K = hazard_logits.shape
    device = hazard_logits.device
    h = torch.sigmoid(hazard_logits).clamp(1e-6, 1 - 1e-6)

    idx = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
    e = event_bin.long().unsqueeze(1)

    # log-survival up to (but not including) the event/censor bin
    surv_mask = (idx < e).float()
    log_surv = (torch.log(1 - h) * surv_mask).sum(dim=1)

    # hazard term at the event bin (events only)
    h_at_e = h.gather(1, e).squeeze(1).clamp(1e-6, 1 - 1e-6)
    event = event_indicator.float()
    log_event = event * torch.log(h_at_e) + (1 - event) * torch.log(1 - h_at_e)

    nll = -(log_surv + log_event)
    return nll.mean()


def cox_ph_loss(log_risk, times, events):
    """Negative Cox partial log-likelihood with Breslow handling of ties.

    loss = - (1 / n_events) * sum_{i: event} [ eta_i - log sum_{j: T_j >= T_i} exp(eta_j) ]

    The risk set is computed over the mini-batch (as in DeepSurv), so the loss
    is only an approximation of the full-cohort partial likelihood; larger
    batches give a better approximation. Tied event times share the same risk
    set (Breslow). Returns 0 (with a graph) when the batch has no events.
    """
    eta = log_risk.reshape(-1)
    t = times.reshape(-1).float()
    e = events.reshape(-1).float()
    if e.sum() <= 0:
        return eta.sum() * 0.0
    order = torch.argsort(t, descending=True)
    eta_s, t_s, e_s = eta[order], t[order], e[order]
    lcse = torch.logcumsumexp(eta_s, dim=0)          # log sum_{j <= i in desc order}
    # Breslow ties: everyone tied at t_i is in the risk set -> use the value at
    # the LAST position of each tie group (descending order keeps ties contiguous).
    _, counts = torch.unique_consecutive(t_s, return_counts=True)
    ends = torch.cumsum(counts, 0) - 1
    group_end = torch.repeat_interleave(ends, counts)
    log_risk_set = lcse[group_end]
    ll = (eta_s - log_risk_set) * e_s
    return -ll.sum() / e_s.sum()
