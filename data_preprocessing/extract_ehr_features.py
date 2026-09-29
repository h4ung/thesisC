"""Extract the structured (EHR) feature vector for each cohort row.

For every (subject_id, study_id, t0) in the cohort, summarise the patient's
history in the baseline lookback window *strictly before t0* into a fixed-length
feature vector for the EHR branch. No information after t0 is used (leakage
control).

Feature groups
--------------
* Demographics: age at t0, sex.
* Vitals / OMR: latest BMI, systolic/diastolic BP, weight (from hosp/omr.csv).
* Labs (last value + min/max/mean/slope in window): creatinine, eGFR, BUN,
  potassium, sodium, bicarbonate, hemoglobin, glucose, albumin, urine ACR if
  available.
* Comorbidity flags from prior diagnoses: diabetes, hypertension, heart failure,
  ischemic heart disease, anemia (these are the classic CKD risk factors).

Outputs:
    dataset/ckd/ehr_features.parquet  (index: subject_id, study_id)
    dataset/ckd/ehr_feature_columns.json
A StandardScaler fit on the TRAIN split only is applied later in the dataloader.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from data_preprocessing.egfr import ckd_epi_2021
from data_preprocessing.build_cohort import real_age, CREATININE_ITEMID

# itemids for common chemistry labs in MIMIC-IV hosp/labevents
LAB_ITEMIDS = {
    "creatinine": 50912,
    "bun": 51006,
    "potassium": 50971,
    "sodium": 50983,
    "bicarbonate": 50882,
    "hemoglobin": 51222,
    "glucose": 50931,
    "albumin": 50862,
}

COMORBIDITY_PREFIXES = {
    # name : (icd9_prefixes, icd10_prefixes)
    "diabetes": (("250",), ("E10", "E11", "E13")),
    "hypertension": (("401", "402", "403", "404", "405"), ("I10", "I11", "I12", "I13", "I15")),
    "heart_failure": (("428",), ("I50",)),
    "ischemic_heart": (("410", "411", "412", "413", "414"), ("I20", "I21", "I22", "I24", "I25")),
    "anemia": (("280", "281", "285"), ("D50", "D51", "D52", "D53", "D63", "D64")),
}


def _slope(times_days, values):
    if len(values) < 2:
        return 0.0
    x = np.asarray(times_days, float)
    y = np.asarray(values, float)
    x = x - x.mean()
    denom = (x ** 2).sum()
    return float((x * (y - y.mean())).sum() / denom) if denom > 0 else 0.0


def _lab_summary(g, t0, prefix):
    """Summary stats of one lab within the window for a single study."""
    out = {}
    if g.empty:
        for s in ("last", "min", "max", "mean", "slope", "n"):
            out[f"{prefix}_{s}"] = np.nan if s != "n" else 0
        return out
    g = g.sort_values("charttime")
    vals = g["valuenum"].to_numpy()
    days = (t0 - g["charttime"]).dt.days.to_numpy() * -1.0
    out[f"{prefix}_last"] = float(vals[-1])
    out[f"{prefix}_min"] = float(np.nanmin(vals))
    out[f"{prefix}_max"] = float(np.nanmax(vals))
    out[f"{prefix}_mean"] = float(np.nanmean(vals))
    out[f"{prefix}_slope"] = _slope(days, vals)
    out[f"{prefix}_n"] = int(len(vals))
    return out


def extract(args):
    os.makedirs(args.out_dir, exist_ok=True)
    cohort = pd.read_parquet(args.cohort)
    cohort["t0"] = pd.to_datetime(cohort["t0"])

    patients = pd.read_csv(args.patients)
    labevents = pd.read_csv(
        args.labevents,
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
    )
    labevents["charttime"] = pd.to_datetime(labevents["charttime"])
    labevents = labevents.dropna(subset=["valuenum"])

    diagnoses = pd.read_csv(args.diagnoses)
    admissions = pd.read_csv(args.admissions)
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    dx = diagnoses.merge(admissions[["hadm_id", "admittime"]], on="hadm_id", how="left")
    dx["code"] = dx["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()

    lookback = pd.to_timedelta(args.baseline_lookback_days, "D")
    rows = []
    # group labs by subject once for speed
    labs_by_subj = {sid: g for sid, g in labevents.groupby("subject_id")}
    dx_by_subj = {sid: g for sid, g in dx.groupby("subject_id")}

    for _, r in cohort.iterrows():
        sid, study_id, t0 = r["subject_id"], r["study_id"], r["t0"]
        feat = {"subject_id": sid, "study_id": study_id}

        # demographics
        feat["age_at_t0"] = float(r["age_at_t0"])
        feat["sex_female"] = 1.0 if str(r.get("gender", "")).upper() == "F" else 0.0

        subj_labs = labs_by_subj.get(sid, labevents.iloc[0:0])
        win = subj_labs[(subj_labs["charttime"] < t0) &
                        (subj_labs["charttime"] >= t0 - lookback)]

        for name, itemid in LAB_ITEMIDS.items():
            feat.update(_lab_summary(win[win["itemid"] == itemid], t0, name))

        # derived eGFR summary from creatinine values in-window
        cr = win[win["itemid"] == CREATININE_ITEMID]
        cr = cr[(cr["valuenum"] > 0.1) & (cr["valuenum"] < 30)]   # add this
        if not cr.empty:
            age = real_age(
                patients.set_index("subject_id").loc[sid, "anchor_age"],
                patients.set_index("subject_id").loc[sid, "anchor_year"],
                cr["charttime"].dt.year,
            )
            egfr_vals = ckd_epi_2021(cr["valuenum"].to_numpy(), np.asarray(age),
                                     feat["sex_female"] == 1.0)
            tmp = cr.assign(valuenum=egfr_vals)
            feat.update(_lab_summary(tmp, t0, "egfr"))
        else:
            feat.update(_lab_summary(cr, t0, "egfr"))

        # comorbidity flags (diagnoses recorded before t0)
        subj_dx = dx_by_subj.get(sid, dx.iloc[0:0])
        prior = subj_dx[subj_dx["admittime"] < t0]
        for cname, (p9, p10) in COMORBIDITY_PREFIXES.items():
            v9 = prior[prior["icd_version"] == 9]["code"]
            v10 = prior[prior["icd_version"] == 10]["code"]
            flag = (
                v9.str.startswith(p9).any() if len(v9) else False
            ) or (
                v10.str.startswith(p10).any() if len(v10) else False
            )
            feat[f"cmb_{cname}"] = 1.0 if flag else 0.0

        rows.append(feat)

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("subject_id", "study_id")]
    out = os.path.join(args.out_dir, "ehr_features.parquet")
    df.to_parquet(out, index=False)
    with open(os.path.join(args.out_dir, "ehr_feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"[ehr] wrote {len(df)} rows x {len(feature_cols)} features -> {out}")
    return df


def get_parser():
    p = argparse.ArgumentParser(description="Extract EHR features for the cohort")
    p.add_argument("--cohort", default="dataset/ckd/ckd_cohort_labels.parquet")
    p.add_argument("--patients", required=True)
    p.add_argument("--admissions", required=True)
    p.add_argument("--diagnoses", required=True)
    p.add_argument("--labevents", required=True)
    p.add_argument("--out_dir", default="dataset/ckd")
    p.add_argument("--baseline_lookback_days", type=int, default=365)
    return p


if __name__ == "__main__":
    extract(get_parser().parse_args())
