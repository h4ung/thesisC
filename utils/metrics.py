"""Evaluation metrics for prognostic CKD models.

Includes standard classification metrics (AUROC, AUPRC, F1 at a threshold,
accuracy) and survival-oriented metrics (Harrell's C-index, time-dependent
AUROC at a chosen horizon bin, and a simple Brier-style calibration error).

scikit-learn is used where available; lightweight fallbacks are provided so the
core training loop never hard-fails on a metric. ``lifelines`` is used for the
C-index when installed, otherwise a vectorised numpy implementation is used.

Thesis C adds ``bootstrap_ci`` (subject-level bootstrap). Survival-specific
utilities (Kaplan-Meier, log-rank, IPCW Brier, calibration) live in
``utils/survival.py``.
"""

import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, accuracy_score,
    )
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False

try:  # optional, faster + independently validated
    from lifelines.utils import concordance_index as _lifelines_cindex
    _HAS_LIFELINES = True
except Exception:  # pragma: no cover
    _HAS_LIFELINES = False


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    if _HAS_SK:
        try:
            out["auroc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            out["auroc"] = float("nan")
        try:
            out["auprc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            out["auprc"] = float("nan")
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        out["acc"] = float(accuracy_score(y_true, y_pred))
    else:
        out["acc"] = float((y_pred == y_true).mean())
    return out


def concordance_index(event_times, predicted_risk, event_observed,
                      use_lifelines=None, block=2048):
    """Harrell's C-index. Higher ``predicted_risk`` should mean an earlier event.

    Comparable pairs are (i, j) with i experiencing the event and t_j > t_i. A
    pair is concordant if r_i > r_j and half-credited if r_i == r_j.

    Implementation
    --------------
    Uses ``lifelines.utils.concordance_index`` when lifelines is installed (it
    takes *risk* here, so the sign is flipped: lifelines expects a score where
    higher == longer survival). Otherwise falls back to a blocked, vectorised
    numpy computation -- same O(n^2) pair count, but done in numpy rather than a
    Python double loop, and streamed in blocks of ``block`` rows so a large test
    set does not allocate an n x n matrix.

    Set ``use_lifelines=False`` to force the numpy path (used by the tests).
    """
    t = np.asarray(event_times, dtype=np.float64).ravel()
    r = np.asarray(predicted_risk, dtype=np.float64).ravel()
    e = np.asarray(event_observed).ravel().astype(bool)

    if len(t) == 0 or not e.any():
        return float("nan")

    if use_lifelines is None:
        use_lifelines = _HAS_LIFELINES
    if use_lifelines and _HAS_LIFELINES:
        try:
            return float(_lifelines_cindex(t, -r, e.astype(int)))
        except (ZeroDivisionError, ValueError):  # pragma: no cover
            return float("nan")

    num = den = 0.0
    idx_events = np.flatnonzero(e)
    for s in range(0, idx_events.size, block):
        i = idx_events[s:s + block]                 # (b,)
        later = t[None, :] > t[i][:, None]          # (b, n) comparable pairs
        den += float(later.sum())
        ri = r[i][:, None]
        num += float((later & (ri > r[None, :])).sum())
        num += 0.5 * float((later & (ri == r[None, :])).sum())
    return float(num / den) if den > 0 else float("nan")


def horizon_eligibility(event_times, event_indicator, horizon_days):
    """Who can be scored at a fixed horizon: had the event by then, or was
    followed at least that far. Subjects censored *before* the horizon have an
    unknown label and must be dropped -- counting them as negatives inflates the
    time-dependent AUROC.

    Mirrors ``eligible_binary`` in data_preprocessing/build_cohort.py; use that
    column when it is available and this helper only as a fallback.
    """
    t = np.asarray(event_times, dtype=np.float64).ravel()
    e = np.asarray(event_indicator).ravel().astype(bool)
    return (e & (t <= horizon_days)) | (t >= horizon_days)


def time_dependent_auroc(cif_at_horizon, label_at_horizon, eligible_mask=None):
    """AUROC of the cumulative incidence prediction at a fixed horizon bin.

    cif_at_horizon   : (N,) predicted CIF = 1 - S(t_horizon)
    label_at_horizon : (N,) 1 if event by horizon, else 0
    eligible_mask    : (N,) bool -- drop subjects censored before the horizon.
                       Strongly recommended: without it, everyone censored early
                       is scored as a negative and the metric is optimistic.
    """
    cif = np.asarray(cif_at_horizon).ravel()
    lab = np.asarray(label_at_horizon).ravel()
    if eligible_mask is not None:
        m = np.asarray(eligible_mask).ravel().astype(bool)
        cif, lab = cif[m], lab[m]
    if not _HAS_SK or len(np.unique(lab)) < 2:
        return float("nan")
    return float(roc_auc_score(lab, cif))


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    return float(np.mean((y_prob - y_true) ** 2))


def bootstrap_ci(metric_fn, cluster_ids, n_boot=200, alpha=0.05, seed=41):
    """Percentile bootstrap CI, resampling *subjects* (clusters) with replacement.

    With ``--policy all`` one subject can contribute several ECG studies, so
    resampling rows would understate the variance. ``metric_fn(indices)`` must
    return the metric on the given row indices; NaN replicates are dropped.
    """
    ids = np.asarray(cluster_ids).ravel()
    uniq, inv = np.unique(ids, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    rows_of = np.split(order, np.cumsum(np.bincount(inv, minlength=uniq.size))[:-1])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, uniq.size, uniq.size)
        ix = np.concatenate([rows_of[k] for k in pick])
        v = metric_fn(ix)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))
