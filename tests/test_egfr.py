import numpy as np

from data_preprocessing.egfr import ckd_epi_2021, ckd_stage


def test_egfr_monotonic_in_creatinine():
    low = ckd_epi_2021(0.8, 50, False)
    high = ckd_epi_2021(2.5, 50, False)
    assert low > high  # higher creatinine -> lower eGFR


def test_egfr_reasonable_range():
    val = float(ckd_epi_2021(1.0, 40, False))
    assert 60 < val < 130  # a healthy 40 y/o male, Scr 1.0 -> normal range


def test_stage_thresholds():
    stages = ckd_stage(np.array([95, 75, 50, 35, 20, 10]))
    assert list(stages) == ["G1", "G2", "G3a", "G3b", "G4", "G5"]


def test_vectorised():
    out = ckd_epi_2021(np.array([0.9, 1.1]), np.array([60, 70]),
                       np.array([True, False]))
    assert out.shape == (2,)
