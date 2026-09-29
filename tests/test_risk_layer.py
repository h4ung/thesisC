"""Thesis C: risk-prediction layer, survival utilities, stratification.

Where lifelines is installed the numpy implementations are cross-checked
against it; otherwise those assertions are skipped.
"""

import numpy as np
import pytest
import torch

from models.heads import BreslowBaseline, CoxRiskHead, horizon_bin_index
from utils.losses import cox_ph_loss
from utils.metrics import bootstrap_ci, concordance_index
from utils.risk_stratification import assign_groups, fit_thresholds, summarise_groups
from utils.survival import (
    administrative_censor, calibration_table, ipcw_brier_score, kaplan_meier, km_at,
    logrank_test,
)

try:
    import lifelines  # noqa: F401
    HAS_LL = True
except Exception:  # pragma: no cover
    HAS_LL = False


def _cohort(n=400, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    beta = np.array([0.9, -0.4])
    t = rng.exponential(1 / np.exp(x @ beta)) * 300
    c = rng.uniform(50, 900, n)
    T = np.minimum(t, c)
    if ties:
        T = np.round(T / 30) * 30 + 1
    E = (t <= c).astype(int)
    return x, beta, T, E


# ---- horizon censoring (the Thesis B labelling bug) ----------------------
def test_events_after_horizon_are_censored_at_horizon():
    t = np.array([100., 700., 800., 2000., 500.])
    e = np.array([1, 1, 1, 1, 0])
    th, eh = administrative_censor(t, e, 730)
    assert list(eh) == [1, 1, 0, 0, 0]          # a year-5 onset is NOT a 2-year event
    assert list(th) == [100., 700., 730., 730., 500.]


def test_dataset_applies_horizon_censoring(tmp_path):
    from data_provider.data_loader import MIMICIV_CKD_Dataset
    from scripts.make_synthetic_dataset import make
    make(str(tmp_path), n_subjects=40, seed=1)
    ds = MIMICIV_CKD_Dataset(str(tmp_path), "train", horizon_days=730, use_ecg=False)
    late = ds.df["event_time_days"] > 730
    assert (ds.df.loc[late, "event_indicator_h"] == 0).all()
    assert (ds.df.loc[late, "event_time_h"] == 730).all()
    item = ds[0]
    assert {"event_indicator_h", "event_time_h", "subject_id", "study_id"} <= set(item)


def test_ehr_group_filter():
    from data_provider.data_loader import filter_ehr_columns
    cols = ["age_at_t0", "sex_female", "egfr_last", "creatinine_min", "bun_n",
            "glucose_last", "cmb_diabetes"]
    assert filter_ehr_columns(cols, "renal") == ["age_at_t0", "sex_female", "glucose_last",
                                                 "cmb_diabetes"]
    assert filter_ehr_columns(cols, "renal,labs_other,comorbidities") == ["age_at_t0", "sex_female"]
    with pytest.raises(ValueError):
        filter_ehr_columns(cols, "demographics,renal,labs_other,comorbidities")


# ---- Kaplan-Meier and log-rank --------------------------------------------
@pytest.mark.skipif(not HAS_LL, reason="lifelines not installed")
def test_kaplan_meier_matches_lifelines():
    from lifelines import KaplanMeierFitter
    _, _, T, E = _cohort(ties=True)
    km = kaplan_meier(T, E)
    kmf = KaplanMeierFitter().fit(T, E)
    q = np.linspace(0, 900, 37)
    assert np.allclose(km_at(km, q), kmf.survival_function_at_times(q).values, atol=1e-10)


@pytest.mark.skipif(not HAS_LL, reason="lifelines not installed")
def test_logrank_matches_lifelines():
    from lifelines.statistics import multivariate_logrank_test
    x, _, T, E = _cohort(ties=True)
    g = (x[:, 0] > 0).astype(int) + (x[:, 1] > 0.5).astype(int)   # 3 groups
    ours = logrank_test(T, E, g)
    ref = multivariate_logrank_test(T, g, E)
    assert ours["statistic"] == pytest.approx(ref.test_statistic, rel=1e-8)
    assert ours["p_value"] == pytest.approx(ref.p_value, rel=1e-6)


# ---- Cox loss and Breslow baseline ----------------------------------------
def test_cox_loss_equals_breslow_partial_likelihood_with_ties():
    x, beta, T, E = _cohort(n=200, ties=True)
    eta = x @ beta
    manual = -np.mean([eta[i] - np.log(np.exp(eta[T >= T[i]]).sum())
                       for i in range(len(T)) if E[i]]) * 1.0
    ours = cox_ph_loss(torch.tensor(eta), torch.tensor(T), torch.tensor(E)).item()
    assert ours == pytest.approx(manual, rel=1e-10)


def test_cox_loss_lower_for_true_coefficients_and_zero_without_events():
    x, beta, T, E = _cohort()
    X, TT, EE = map(torch.tensor, (x, T, E))
    good = cox_ph_loss(X @ torch.tensor(beta), TT, EE)
    bad = cox_ph_loss(-(X @ torch.tensor(beta)), TT, EE)
    assert good < bad
    zero = cox_ph_loss(torch.randn(5, requires_grad=True), torch.rand(5), torch.zeros(5))
    assert zero.item() == 0.0 and zero.requires_grad


@pytest.mark.skipif(not HAS_LL, reason="lifelines not installed")
def test_breslow_absolute_risk_matches_lifelines():
    import pandas as pd
    from lifelines import CoxPHFitter
    x, _, T, E = _cohort(n=300)          # continuous times -> no ties
    df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "T": T, "E": E})
    cph = CoxPHFitter(baseline_estimation_method="breslow").fit(df, "T", "E")
    eta = x @ cph.params_.values
    bl = BreslowBaseline().fit(eta, T, E)
    # compare at observed event times: Breslow is a step function, whereas
    # lifelines linearly interpolates between event times
    ev = np.sort(T[E == 1])
    times = [ev[10], ev[50], ev[100]]
    ours = bl.abs_risk(eta, times)
    ref = 1 - cph.predict_survival_function(df[["a", "b"]], times=times).values.T
    assert np.allclose(ours, ref, atol=1e-6)
    # round-trip through state_dict
    bl2 = BreslowBaseline().load_state_dict(bl.state_dict())
    assert np.allclose(bl2.abs_risk(eta, times), ours)


def test_cox_head_and_linear_head_shapes():
    h = CoxRiskHead(16)
    assert h(torch.randn(4, 16)).shape == (4,)
    lin = CoxRiskHead(16, hidden=0)
    assert isinstance(lin.net, torch.nn.Linear)


def test_horizon_bin_index():
    assert horizon_bin_index(730, 730, 8) == 7
    assert horizon_bin_index(365, 730, 8) == 3      # right edge of bin 3 = 365 d
    with pytest.raises(ValueError):
        horizon_bin_index(1000, 730, 8)


# ---- calibration / Brier --------------------------------------------------
def test_ipcw_brier_without_censoring_equals_plain_brier():
    rng = np.random.default_rng(3)
    T = rng.uniform(1, 1000, 500)
    E = np.ones(500, dtype=int)                    # nobody censored
    r = rng.random(500)
    y = (T <= 730).astype(float)
    assert ipcw_brier_score(T, E, r, 730) == pytest.approx(np.mean((r - y) ** 2), rel=1e-9)


def test_ipcw_brier_zero_for_perfect_prediction():
    _, _, T, E = _cohort()
    y = ((E == 1) & (T <= 365)).astype(float)
    assert ipcw_brier_score(T, E, y, 365) == pytest.approx(0.0, abs=1e-12)


def test_calibration_table_counts():
    _, _, T, E = _cohort()
    r = np.random.default_rng(0).random(len(T))
    rows, ece = calibration_table(T, E, r, 365, n_bins=10)
    assert sum(row["n"] for row in rows) == len(T)
    assert 0 <= ece <= 1


# ---- stratification --------------------------------------------------------
def test_quantile_thresholds_come_from_validation_and_split_test():
    val = np.linspace(0, 1, 101)
    th = fit_thresholds(val, "quantile", quantiles="0.5")
    assert th["thresholds"] == [pytest.approx(0.5)]
    g = assign_groups(np.array([0.1, 0.49, 0.51, 0.9]), th["thresholds"])
    assert list(g) == [0, 0, 1, 1]
    th3 = fit_thresholds(val, "quantile", quantiles="0.33,0.67")
    assert list(assign_groups(np.array([0.1, 0.5, 0.9]), th3["thresholds"])) == [0, 1, 2]


def test_youden_and_sensitivity_thresholds():
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    y = np.array([0, 0, 0, 1, 1, 1])
    th = fit_thresholds(s, "youden", val_label=y)
    assert 0.3 < th["thresholds"][0] <= 0.6 and th["val_sensitivity"] == 1.0
    ths = fit_thresholds(s, "sensitivity", val_label=y, target_sensitivity=0.66)
    assert ths["val_sensitivity"] >= 0.66


def test_summarise_groups_separates_high_and_low_risk():
    x, _, T, E = _cohort(n=600)
    T, E = administrative_censor(T, E, 730)
    risk = x[:, 0]                                     # true dominant risk factor
    g = assign_groups(risk, [np.median(risk)])
    s = summarise_groups(T, E, g, horizon=730)
    low, high = s["groups"]
    assert high["observed_risk_km"] > low["observed_risk_km"]
    assert s["logrank_p"] < 1e-3 and s["hr_high_vs_low"] > 1.5


def test_bootstrap_ci_brackets_point_estimate():
    _, beta, T, E = _cohort()
    x, _, _, _ = _cohort()
    r = x @ beta
    ids = np.arange(len(T)) // 2                        # 2 rows per "subject"
    lo, hi = bootstrap_ci(lambda ix: concordance_index(T[ix], r[ix], E[ix]), ids, n_boot=50)
    c = concordance_index(T, r, E)
    assert lo < c < hi
