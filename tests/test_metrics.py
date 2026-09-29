"""Metric regression tests: C-index equivalence, and censoring-aware td-AUROC."""

import numpy as np

from utils.metrics import (
    concordance_index, horizon_eligibility, time_dependent_auroc,
)


def _naive_cindex(t, r, e):
    """The original O(n^2) Python double loop, kept here as ground truth."""
    t, r, e = np.asarray(t), np.asarray(r), np.asarray(e).astype(bool)
    num = den = 0.0
    for i in range(len(t)):
        if not e[i]:
            continue
        for j in range(len(t)):
            if t[j] > t[i]:
                den += 1
                if r[i] > r[j]:
                    num += 1
                elif r[i] == r[j]:
                    num += 0.5
    return num / den if den else float("nan")


def test_vectorised_cindex_matches_naive_loop():
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = 200
        t = rng.integers(1, 40, n).astype(float)   # ties in time
        r = np.round(rng.random(n), 2)             # ties in risk
        e = rng.random(n) > 0.7
        got = concordance_index(t, r, e, use_lifelines=False)
        assert abs(got - _naive_cindex(t, r, e)) < 1e-9


def test_cindex_blocking_does_not_change_result():
    rng = np.random.default_rng(1)
    t, r, e = rng.random(500) * 100, rng.random(500), rng.random(500) > 0.5
    a = concordance_index(t, r, e, use_lifelines=False, block=7)
    b = concordance_index(t, r, e, use_lifelines=False, block=10_000)
    assert abs(a - b) < 1e-9


def test_cindex_perfect_and_inverted():
    t = np.array([10.0, 20.0, 30.0, 40.0])
    e = np.array([1, 1, 1, 0])
    perfect = np.array([0.9, 0.8, 0.7, 0.6])   # higher risk -> earlier event
    assert concordance_index(t, perfect, e, use_lifelines=False) == 1.0
    assert concordance_index(t, -perfect, e, use_lifelines=False) == 0.0


def test_cindex_nan_without_events():
    t = np.arange(5.0)
    assert np.isnan(concordance_index(t, np.random.rand(5), np.zeros(5), use_lifelines=False))


def test_eligibility_excludes_early_censored():
    horizon = 730.0
    times = np.array([100.0, 300.0, 800.0, 730.0])
    events = np.array([1, 0, 0, 0])           # #2 is censored at day 300 -> unknown label
    elig = horizon_eligibility(times, events, horizon)
    assert list(elig) == [True, False, True, True]


def test_td_auroc_is_inflated_when_censored_counted_as_negative():
    """Patients censored before the horizon have an unknown label. Scoring them
    as negatives (eligible_mask=None) flatters the model -- that is the bug."""
    horizon = 730.0
    # 4 real positives, 4 followed-to-horizon negatives, 6 censored at day 50
    times = np.array([100, 200, 300, 400] + [800] * 4 + [50] * 6, dtype=float)
    events = np.array([1, 1, 1, 1] + [0] * 4 + [0] * 6)
    labels = np.array([1, 1, 1, 1] + [0] * 4 + [0] * 6)
    # the model separates the real positives from the real negatives only so-so,
    # but gives the early-censored patients very low risk -- they pad the negative
    # class with free wins the model was never actually tested on.
    risk = np.array([0.9, 0.8, 0.45, 0.4] + [0.5] * 4 + [0.01] * 6)

    elig = horizon_eligibility(times, events, horizon)
    honest = time_dependent_auroc(risk, labels, eligible_mask=elig)
    inflated = time_dependent_auroc(risk, labels, eligible_mask=None)
    assert elig.sum() == 8
    assert honest == 0.5          # the model is actually a coin flip here
    assert inflated > honest      # ... but scores 0.8 if early-censored count as 0s
