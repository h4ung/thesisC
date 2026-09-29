"""Risk stratification: turn a continuous predicted risk into patient risk groups.

The prognostic layer outputs a *continuous* risk for every patient (a predicted
probability of incident CKD by the horizon, and a ranking score). Clinically the
question is usually "who is high risk and who is low risk?". This module answers
it without leaking test information:

1. **Thresholds are fitted on the validation split only** (``fit_thresholds``)
   and then frozen.
2. The frozen thresholds are applied to the test split (``assign_groups``).
3. Each group is summarised with the *observed* (Kaplan-Meier) CKD incidence,
   and the groups are compared with a log-rank test and a hazard ratio
   (``summarise_groups``).

Supported methods (``--risk_strat_method``)
-------------------------------------------
quantile     cut the validation risk distribution at the given quantiles.
             ``--risk_strat_quantiles 0.5``      -> low / high (median split)
             ``--risk_strat_quantiles 0.33,0.67`` -> low / intermediate / high
             ``--risk_strat_quantiles 0.8``      -> top-20% flagged as high risk
absolute     cut predicted *absolute* risk at the horizon at fixed probabilities,
             e.g. ``--risk_strat_cutoffs 0.05,0.15`` (clinically interpretable:
             "<5% two-year risk is low").
youden       one cut-point on validation maximising sensitivity + specificity - 1
             for "CKD by the horizon" among horizon-eligible patients.
sensitivity  the highest cut-point that still reaches ``--risk_strat_target_sens``
             sensitivity on validation (screening-style operating point).
"""

import numpy as np

from utils.survival import (
    cox_hazard_ratio, kaplan_meier, km_at, km_risk_at, logrank_test,
)

METHODS = ("quantile", "absolute", "youden", "sensitivity")


def _parse_floats(s):
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [float(x) for x in s]
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def group_names(n_groups):
    if n_groups == 2:
        return ["low", "high"]
    if n_groups == 3:
        return ["low", "intermediate", "high"]
    return [f"Q{i + 1}" for i in range(n_groups)]


def fit_thresholds(val_score, method="quantile", quantiles="0.5", cutoffs="0.05,0.15",
                   val_label=None, val_eligible=None, target_sensitivity=0.8,
                   val_abs_risk=None):
    """Fit cut-points on the *validation* split.

    Returns dict(method, thresholds, on) where ``on`` says which array the
    thresholds apply to: 'score' (ranking score) or 'abs_risk' (absolute risk at
    the stratification horizon).
    """
    s = np.asarray(val_score, dtype=float).ravel()
    if method not in METHODS:
        raise ValueError(f"unknown risk_strat_method {method!r}; choose from {METHODS}")

    if method == "quantile":
        qs = sorted(_parse_floats(quantiles))
        if not qs or any(q <= 0 or q >= 1 for q in qs):
            raise ValueError("risk_strat_quantiles must be in (0, 1)")
        th = np.quantile(s, qs).tolist()
        return {"method": method, "thresholds": th, "on": "score", "quantiles": qs}

    if method == "absolute":
        cs = sorted(_parse_floats(cutoffs))
        if val_abs_risk is None:
            raise ValueError("absolute stratification needs absolute risk at the horizon")
        return {"method": method, "thresholds": cs, "on": "abs_risk"}

    # label-based single cut-points ----------------------------------------
    if val_label is None:
        raise ValueError(f"{method} stratification needs validation labels")
    y = np.asarray(val_label).ravel().astype(int)
    m = np.ones_like(y, dtype=bool) if val_eligible is None else np.asarray(val_eligible).ravel().astype(bool)
    s_m, y_m = s[m], y[m]
    if y_m.min(initial=1) == y_m.max(initial=0):
        # only one class present -> fall back to the median
        return {"method": "quantile", "thresholds": [float(np.median(s))], "on": "score",
                "note": f"{method} fell back to median split (single class on validation)"}
    cands = np.unique(s_m)
    P, N = (y_m == 1).sum(), (y_m == 0).sum()
    # vectorised sens/spec for "score >= c is high risk"
    order = np.sort(s_m[y_m == 1]); order_n = np.sort(s_m[y_m == 0])
    sens = 1 - np.searchsorted(order, cands, side="left") / P
    spec = np.searchsorted(order_n, cands, side="left") / N
    if method == "youden":
        j = sens + spec - 1
        c = float(cands[int(np.argmax(j))])
        return {"method": method, "thresholds": [c], "on": "score",
                "val_sensitivity": float(sens[np.argmax(j)]),
                "val_specificity": float(spec[np.argmax(j)])}
    # sensitivity target: largest threshold still achieving the target
    ok = np.where(sens >= target_sensitivity)[0]
    i = int(ok.max()) if ok.size else 0
    return {"method": method, "thresholds": [float(cands[i])], "on": "score",
            "val_sensitivity": float(sens[i]), "val_specificity": float(spec[i]),
            "target_sensitivity": float(target_sensitivity)}


def assign_groups(values, thresholds):
    """Group index 0..len(thresholds): 0 = lowest risk. ``values >= t`` moves up."""
    v = np.asarray(values, dtype=float).ravel()
    return np.searchsorted(np.asarray(sorted(thresholds)), v, side="right").astype(int)


def summarise_groups(times, events, groups, abs_risk=None, horizon=730.0, names=None,
                     km_grid=None):
    """Per-group observed vs predicted risk, log-rank test and HR (top vs bottom).

    ``times``/``events`` should already be administratively censored at the
    horizon (see utils.survival.administrative_censor).
    """
    t = np.asarray(times, dtype=float).ravel()
    e = np.asarray(events).ravel().astype(int)
    g = np.asarray(groups).ravel().astype(int)
    n_groups = int(max(g.max(initial=0) + 1, len(names) if names else 0))
    names = names or group_names(n_groups)

    rows = []
    for k in range(n_groups):
        m = g == k
        obs, lo, hi = km_risk_at(t[m], e[m], horizon) if m.any() else (np.nan,) * 3
        rows.append({
            "group": names[k] if k < len(names) else f"G{k}",
            "n": int(m.sum()),
            "fraction": float(m.mean()) if m.size else float("nan"),
            "n_events_by_horizon": int((e[m] == 1).sum()),
            "observed_risk_km": float(obs), "observed_risk_lo": float(lo),
            "observed_risk_hi": float(hi),
            "mean_predicted_risk": (float(np.mean(np.asarray(abs_risk)[m]))
                                    if abs_risk is not None and m.any() else float("nan")),
        })

    present = np.unique(g)
    lr = logrank_test(t, e, g) if present.size >= 2 else {"statistic": np.nan, "p_value": np.nan, "dof": 0}
    top, bottom = int(present.max()), int(present.min())
    m2 = (g == top) | (g == bottom)
    hr, hr_lo, hr_hi, hr_method = (cox_hazard_ratio(t[m2], e[m2], (g[m2] == top))
                                   if present.size >= 2 else (np.nan,) * 3 + ("n/a",))

    # KM curves per group on a common grid, for plotting
    grid = np.asarray(km_grid if km_grid is not None else np.linspace(0, horizon, 49))
    curves = {}
    for k in range(n_groups):
        m = g == k
        if m.any():
            km = kaplan_meier(t[m], e[m])
            curves[rows[k]["group"]] = (1 - km_at(km, grid)).tolist()   # cumulative incidence

    top_obs = rows[top]["observed_risk_km"]
    bot_obs = rows[bottom]["observed_risk_km"]
    return {
        "groups": rows,
        "logrank_statistic": float(lr["statistic"]),
        "logrank_dof": int(lr["dof"]),
        "logrank_p": float(lr["p_value"]),
        "hr_high_vs_low": float(hr), "hr_lo": float(hr_lo), "hr_hi": float(hr_hi),
        "hr_method": hr_method,
        "observed_risk_ratio_high_vs_low": (float(top_obs / bot_obs)
                                            if bot_obs and np.isfinite(bot_obs) and bot_obs > 0
                                            else float("nan")),
        "km_grid_days": grid.tolist(),
        "km_cumulative_incidence": curves,
    }
