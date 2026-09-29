"""Survival-analysis utilities for the risk-prediction layer (Thesis C).

Everything here is plain numpy (scipy only for the chi-square tail), so it is
unit-testable without torch and without lifelines. When lifelines *is* installed
the tests cross-check these implementations against it.

Contents
--------
administrative_censor   truncate follow-up at the prediction horizon
kaplan_meier            product-limit estimator with Greenwood variance
km_at                   step-function lookup of a KM curve at arbitrary times
logrank_test            k-group log-rank test (chi-square, k-1 dof)
hazard_ratio_logrank    HR of one group vs another from log-rank O/E
ipcw_brier_score        Graf et al. (1999) Brier score at a horizon with
                        inverse-probability-of-censoring weights
calibration_table       predicted vs KM-observed risk by risk quantile
"""

import numpy as np

try:
    from scipy.stats import chi2 as _chi2
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def _as1d(x, dtype=np.float64):
    return np.asarray(x, dtype=dtype).ravel()


# --------------------------------------------------------------------------
# censoring at the horizon
# --------------------------------------------------------------------------
def administrative_censor(times, events, horizon):
    """Censor everyone still event-free at ``horizon``.

    An event observed *after* the horizon is not an event within the prediction
    window: the patient is known to be event-free up to the horizon, so they are
    censored there. Without this, a CKD onset at year 5 is counted as a year-2
    event by any model that bins time over [0, horizon].
    """
    t = _as1d(times)
    e = _as1d(events).astype(bool)
    after = t > horizon
    t_h = np.where(after, float(horizon), t)
    e_h = e & ~after
    return t_h, e_h.astype(int)


# --------------------------------------------------------------------------
# Kaplan-Meier
# --------------------------------------------------------------------------
def kaplan_meier(times, events):
    """Kaplan-Meier survival estimate.

    Returns dict with arrays over the distinct event/censor times:
        time, n_at_risk, n_events, surv, var (Greenwood variance of S).
    ``surv`` is the value *after* the drop at ``time`` (right-continuous).
    """
    t = _as1d(times)
    e = _as1d(events).astype(bool)
    if t.size == 0:
        return {"time": np.array([]), "n_at_risk": np.array([]),
                "n_events": np.array([]), "surv": np.array([]), "var": np.array([])}
    uniq = np.unique(t)
    # at-risk: T >= u ; events: T == u & e
    order = np.sort(t)
    n_at_risk = t.size - np.searchsorted(order, uniq, side="left")
    ev_times = np.sort(t[e])
    n_events = (np.searchsorted(ev_times, uniq, side="right")
                - np.searchsorted(ev_times, uniq, side="left"))
    with np.errstate(divide="ignore", invalid="ignore"):
        step = 1.0 - n_events / np.maximum(n_at_risk, 1)
        surv = np.cumprod(step)
        gw_terms = np.where(
            (n_at_risk - n_events) > 0,
            n_events / (n_at_risk * np.maximum(n_at_risk - n_events, 1)),
            0.0,
        )
    var = surv ** 2 * np.cumsum(gw_terms)
    return {"time": uniq, "n_at_risk": n_at_risk, "n_events": n_events,
            "surv": surv, "var": var}


def km_at(km, query_times, left_limit=False):
    """Evaluate a KM curve at ``query_times``.

    ``left_limit=True`` returns S(t-) (the value just *before* t), which is what
    the IPCW weights need for event times.
    """
    q = _as1d(query_times)
    if km["time"].size == 0:
        return np.ones_like(q)
    side = "left" if left_limit else "right"
    idx = np.searchsorted(km["time"], q, side=side) - 1
    out = np.ones_like(q)
    ok = idx >= 0
    out[ok] = km["surv"][idx[ok]]
    return out


def km_risk_at(times, events, horizon):
    """Observed cumulative incidence 1 - S(horizon) with a 95% CI (log-log)."""
    km = kaplan_meier(times, events)
    if km["time"].size == 0:
        return float("nan"), float("nan"), float("nan")
    s = float(km_at(km, [horizon])[0])
    idx = np.searchsorted(km["time"], horizon, side="right") - 1
    var = float(km["var"][idx]) if idx >= 0 else 0.0
    if s <= 0 or s >= 1 or var <= 0:
        return 1 - s, 1 - s, 1 - s
    # log(-log) transformed CI for S, mapped to risk = 1 - S
    se = np.sqrt(var) / (s * abs(np.log(s)))
    lo_s = s ** np.exp(1.96 * se)
    hi_s = s ** np.exp(-1.96 * se)
    return 1 - s, 1 - hi_s, 1 - lo_s


# --------------------------------------------------------------------------
# log-rank test and hazard ratio
# --------------------------------------------------------------------------
def logrank_test(times, events, groups):
    """k-sample log-rank test.

    Returns dict(statistic, dof, p_value, observed, expected) where observed and
    expected are per-group event counts (sorted by group label).
    """
    t = _as1d(times)
    e = _as1d(events).astype(bool)
    g = np.asarray(groups).ravel()
    labels = np.unique(g)
    k = labels.size
    if k < 2 or not e.any():
        return {"statistic": float("nan"), "dof": max(k - 1, 0), "p_value": float("nan"),
                "observed": [0.0] * k, "expected": [0.0] * k, "groups": labels.tolist()}

    ev_times = np.unique(t[e])
    O = np.zeros(k)
    E = np.zeros(k)
    V = np.zeros((k, k))
    for u in ev_times:
        at_risk = t >= u
        d_mask = (t == u) & e
        n = at_risk.sum()
        d = d_mask.sum()
        n_g = np.array([(at_risk & (g == lab)).sum() for lab in labels], dtype=float)
        d_g = np.array([(d_mask & (g == lab)).sum() for lab in labels], dtype=float)
        O += d_g
        E += d * n_g / n
        if n > 1:
            c = d * (n - d) / (n ** 2 * (n - 1))
            V += c * (np.diag(n_g) * n - np.outer(n_g, n_g))
    # drop last group to get a non-singular (k-1)x(k-1) system
    diff = (O - E)[:-1]
    Vr = V[:-1, :-1]
    try:
        stat = float(diff @ np.linalg.solve(Vr, diff))
    except np.linalg.LinAlgError:
        stat = float(diff @ np.linalg.pinv(Vr) @ diff)
    dof = k - 1
    p = float(_chi2.sf(stat, dof)) if _HAS_SCIPY else float("nan")
    return {"statistic": stat, "dof": dof, "p_value": p,
            "observed": O.tolist(), "expected": E.tolist(), "groups": labels.tolist()}


def hazard_ratio_logrank(times, events, groups, group_a, group_b):
    """Hazard ratio of ``group_a`` relative to ``group_b`` via log-rank O/E.

    HR = (O_a / E_a) / (O_b / E_b), computed on the two groups only, with an
    approximate 95% CI from Var(log HR) = 1/E_a + 1/E_b. This is the classical
    Peto estimate; for publication-grade numbers prefer a Cox fit (lifelines),
    which ``cox_hazard_ratio`` uses when available.
    """
    g = np.asarray(groups).ravel()
    m = (g == group_a) | (g == group_b)
    res = logrank_test(_as1d(times)[m], _as1d(events)[m], g[m])
    labels = res["groups"]
    if len(labels) < 2:
        return float("nan"), float("nan"), float("nan")
    ia, ib = labels.index(group_a), labels.index(group_b)
    O, E = np.array(res["observed"]), np.array(res["expected"])
    if min(O[ia], O[ib], E[ia], E[ib]) <= 0:
        return float("nan"), float("nan"), float("nan")
    hr = (O[ia] / E[ia]) / (O[ib] / E[ib])
    se = np.sqrt(1.0 / E[ia] + 1.0 / E[ib])
    return float(hr), float(hr * np.exp(-1.96 * se)), float(hr * np.exp(1.96 * se))


def cox_hazard_ratio(times, events, is_group_a):
    """HR (group A vs rest) from a univariable Cox model if lifelines is installed,
    else the log-rank estimate. Returns (hr, lo, hi, method)."""
    t = _as1d(times)
    e = _as1d(events).astype(int)
    a = np.asarray(is_group_a).astype(int).ravel()
    try:
        import pandas as pd
        from lifelines import CoxPHFitter
        df = pd.DataFrame({"T": t, "E": e, "A": a})
        cph = CoxPHFitter().fit(df, "T", "E")
        s = cph.summary.loc["A"]
        return (float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]),
                float(s["exp(coef) upper 95%"]), "cox")
    except Exception:
        hr, lo, hi = hazard_ratio_logrank(t, e, a, 1, 0)
        return hr, lo, hi, "logrank"


# --------------------------------------------------------------------------
# calibration / accuracy of absolute risk
# --------------------------------------------------------------------------
def ipcw_brier_score(times, events, risk_at_h, horizon, train_times=None, train_events=None):
    """Brier score of predicted cumulative incidence at ``horizon`` with IPCW.

    Graf et al. (1999): subjects censored before the horizon get weight 0 and
    the others are re-weighted by 1/G, where G is the KM estimate of the
    *censoring* distribution (fit on ``train_*`` if given, else on the data).

        event by h      : (1 - r_i)^2 / G(T_i-)
        survived past h :      r_i ^2 / G(h)
        censored before : 0
    """
    t = _as1d(times)
    e = _as1d(events).astype(bool)
    r = _as1d(risk_at_h)
    ct = _as1d(train_times) if train_times is not None else t
    ce = _as1d(train_events).astype(bool) if train_events is not None else e
    G = kaplan_meier(ct, ~ce)                         # censoring "survival"
    g_ti = np.clip(km_at(G, t, left_limit=True), 1e-8, None)
    g_h = max(float(km_at(G, [horizon])[0]), 1e-8)

    case = e & (t <= horizon)
    ctrl = t > horizon
    contrib = np.zeros_like(r)
    contrib[case] = (1 - r[case]) ** 2 / g_ti[case]
    contrib[ctrl] = r[ctrl] ** 2 / g_h
    return float(contrib.mean()) if contrib.size else float("nan")


def calibration_table(times, events, risk_at_h, horizon, n_bins=10):
    """Mean predicted risk vs KM-observed risk within quantile bins of risk.

    Returns a list of dicts (one per bin) and the expected-calibration error
    ``ece = sum_b (n_b / N) * |pred_b - obs_b|``. The KM estimate inside each
    bin accounts for censoring, unlike a naive event fraction.
    """
    t = _as1d(times)
    e = _as1d(events).astype(bool)
    r = _as1d(risk_at_h)
    n = r.size
    if n == 0:
        return [], float("nan")
    n_bins = int(max(1, min(n_bins, n)))
    edges = np.quantile(r, np.linspace(0, 1, n_bins + 1))
    bin_id = np.clip(np.searchsorted(edges[1:-1], r, side="right"), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = bin_id == b
        if not m.any():
            continue
        obs, lo, hi = km_risk_at(t[m], e[m], horizon)
        pred = float(r[m].mean())
        rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": pred,
                     "observed_km": float(obs), "observed_lo": float(lo),
                     "observed_hi": float(hi), "n_events": int((e[m] & (t[m] <= horizon)).sum())})
        if np.isfinite(obs):
            ece += (m.sum() / n) * abs(pred - obs)
    return rows, float(ece)
