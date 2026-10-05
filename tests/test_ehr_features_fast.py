"""The fast extract_ehr_features must give exactly the Thesis B per-study output."""

import argparse
import os

import numpy as np
import pandas as pd

from data_preprocessing import extract_ehr_features as fast
from data_preprocessing.extract_ehr_features import (
    CREATININE_ITEMID, COMORBIDITY_PREFIXES, LAB_ITEMIDS, _lab_summary, ckd_epi_2021, real_age,
)


def reference_extract(args):
    """The original Thesis B loop, kept verbatim (minus I/O) as ground truth."""
    cohort = pd.read_parquet(args.cohort); cohort["t0"] = pd.to_datetime(cohort["t0"])
    patients = pd.read_csv(args.patients)
    labevents = pd.read_csv(args.labevents, usecols=["subject_id", "itemid", "charttime", "valuenum"])
    labevents["charttime"] = pd.to_datetime(labevents["charttime"])
    labevents = labevents.dropna(subset=["valuenum"])
    diagnoses = pd.read_csv(args.diagnoses); admissions = pd.read_csv(args.admissions)
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    dx = diagnoses.merge(admissions[["hadm_id", "admittime"]], on="hadm_id", how="left")
    dx["code"] = dx["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()
    lookback = pd.to_timedelta(args.baseline_lookback_days, "D")
    labs_by_subj = {sid: g for sid, g in labevents.groupby("subject_id")}
    dx_by_subj = {sid: g for sid, g in dx.groupby("subject_id")}
    rows = []
    for _, r in cohort.iterrows():
        sid, t0 = r["subject_id"], r["t0"]
        feat = {"subject_id": sid, "study_id": r["study_id"], "age_at_t0": float(r["age_at_t0"]),
                "sex_female": 1.0 if str(r.get("gender", "")).upper() == "F" else 0.0}
        sl = labs_by_subj.get(sid, labevents.iloc[0:0])
        win = sl[(sl["charttime"] < t0) & (sl["charttime"] >= t0 - lookback)]
        for name, itemid in LAB_ITEMIDS.items():
            feat.update(_lab_summary(win[win["itemid"] == itemid], t0, name))
        cr = win[win["itemid"] == CREATININE_ITEMID]
        cr = cr[(cr["valuenum"] > 0.1) & (cr["valuenum"] < 30)]
        if not cr.empty:
            age = real_age(patients.set_index("subject_id").loc[sid, "anchor_age"],
                           patients.set_index("subject_id").loc[sid, "anchor_year"], cr["charttime"].dt.year)
            eg = ckd_epi_2021(cr["valuenum"].to_numpy(), np.asarray(age), feat["sex_female"] == 1.0)
            feat.update(_lab_summary(cr.assign(valuenum=eg), t0, "egfr"))
        else:
            feat.update(_lab_summary(cr, t0, "egfr"))
        sd = dx_by_subj.get(sid, dx.iloc[0:0]); prior = sd[sd["admittime"] < t0]
        for cname, (p9, p10) in COMORBIDITY_PREFIXES.items():
            v9 = prior[prior["icd_version"] == 9]["code"]; v10 = prior[prior["icd_version"] == 10]["code"]
            flag = (v9.str.startswith(p9).any() if len(v9) else False) or \
                   (v10.str.startswith(p10).any() if len(v10) else False)
            feat[f"cmb_{cname}"] = 1.0 if flag else 0.0
        rows.append(feat)
    return pd.DataFrame(rows)


def _fake_mimic(d, n_subj=60, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2150-01-01")
    pats = pd.DataFrame({"subject_id": range(1, n_subj + 1), "gender": rng.choice(["F", "M"], n_subj),
                         "anchor_age": rng.integers(20, 80, n_subj), "anchor_year": 2150})
    pats.to_csv(d / "patients.csv.gz", index=False)
    items = list(LAB_ITEMIDS.values()) + [99999]
    m = 20000
    labs = pd.DataFrame({"subject_id": rng.integers(1, n_subj + 3, m),     # some not in cohort
                         "hadm_id": rng.integers(1, 300, m), "itemid": rng.choice(items, m),
                         "charttime": base + pd.to_timedelta(rng.integers(0, 3 * 365 * 24 * 60, m), "min"),
                         "valuenum": np.where(rng.random(m) < 0.05, np.nan, rng.gamma(2, 1, m))})
    labs.loc[labs.itemid == CREATININE_ITEMID, "valuenum"] *= 0.5
    labs.loc[rng.random(m) < 0.01, "valuenum"] = 40.0                       # out-of-range creatinine
    labs.to_csv(d / "labevents.csv.gz", index=False)
    adm = pd.DataFrame({"hadm_id": range(1, 300), "subject_id": rng.integers(1, n_subj + 1, 299),
                        "admittime": base + pd.to_timedelta(rng.integers(0, 3 * 365, 299), "D")})
    adm.to_csv(d / "admissions.csv.gz", index=False)
    codes = [("2500", 9), ("4019", 9), ("E119", 10), ("I10", 10), ("I509", 10), ("D649", 10),
             ("4280", 9), ("Z000", 10), ("V700", 9), ("I251", 10), ("2851", 9)]
    k = 900
    pick = rng.integers(0, len(codes), k)
    dxt = pd.DataFrame({"subject_id": rng.integers(1, n_subj + 1, k), "hadm_id": rng.integers(1, 320, k),
                        "icd_code": [codes[i][0] for i in pick], "icd_version": [codes[i][1] for i in pick]})
    dxt.to_csv(d / "diagnoses_icd.csv.gz", index=False)
    nc = 150
    sid = rng.integers(1, n_subj + 1, nc)
    coh = pd.DataFrame({"subject_id": sid, "study_id": range(10_000, 10_000 + nc),
                        "t0": base + pd.to_timedelta(rng.integers(100, 3 * 365 * 24 * 60, nc), "min"),
                        "age_at_t0": rng.uniform(20, 80, nc),
                        "gender": pats.set_index("subject_id").loc[sid, "gender"].to_numpy()})
    coh.to_parquet(d / "cohort.parquet", index=False)


def test_fast_extract_matches_reference(tmp_path):
    _fake_mimic(tmp_path)
    args = argparse.Namespace(cohort=str(tmp_path / "cohort.parquet"), patients=str(tmp_path / "patients.csv.gz"),
                              admissions=str(tmp_path / "admissions.csv.gz"),
                              diagnoses=str(tmp_path / "diagnoses_icd.csv.gz"),
                              labevents=str(tmp_path / "labevents.csv.gz"), out_dir=str(tmp_path / "out"),
                              baseline_lookback_days=365)
    ref = reference_extract(args)
    new = fast.extract(args)
    assert list(new.columns) == list(ref.columns)
    assert len(new) == len(ref)
    pd.testing.assert_frame_equal(new.reset_index(drop=True), ref.reset_index(drop=True),
                                  check_dtype=False, rtol=1e-9, atol=1e-12)
    assert os.path.exists(tmp_path / "out" / "ehr_feature_columns.json")
    # the comparison is not vacuous: windows, eGFR and flags are populated
    assert ref["creatinine_n"].gt(0).any() and ref["egfr_n"].gt(0).any() and ref["cmb_diabetes"].gt(0).any()
