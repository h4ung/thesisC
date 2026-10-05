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


def _window_summary(times_ns, vals, t0_ns, lo_ns, prefix):
    """Same statistics as _lab_summary, on pre-sorted numpy arrays.

    Window = [t0 - lookback, t0): lo inclusive, t0 exclusive (as before).
    """
    i0 = np.searchsorted(times_ns, lo_ns, side="left")
    i1 = np.searchsorted(times_ns, t0_ns, side="left")
    if i1 <= i0:
        return {f"{prefix}_last": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan,
                f"{prefix}_mean": np.nan, f"{prefix}_slope": np.nan, f"{prefix}_n": 0}
    v = vals[i0:i1]
    # (t0 - charttime).dt.days floors to whole days; negated = days before t0
    days = -((t0_ns - times_ns[i0:i1]) // _NS_PER_DAY).astype(float)
    return {f"{prefix}_last": float(v[-1]), f"{prefix}_min": float(np.nanmin(v)),
            f"{prefix}_max": float(np.nanmax(v)), f"{prefix}_mean": float(np.nanmean(v)),
            f"{prefix}_slope": _slope(days, v), f"{prefix}_n": int(len(v))}


_NS_PER_DAY = 86_400 * 10**9


def extract(args):
    """Vectorised-per-subject version (Thesis C).

    The Thesis B loop rebuilt patients.set_index() twice per ECG study and ran
    ~20 pandas filters per study, which takes many hours on the full cohort.
    Here each subject's labs are sorted once into numpy arrays per lab test,
    windows are found with binary search, and comorbidity flags reduce to
    "earliest qualifying admission < t0". Output is identical (see
    tests/test_ehr_features_fast.py), and progress is printed as it runs.
    """
    import time
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()
    cohort = pd.read_parquet(args.cohort)
    cohort["t0"] = pd.to_datetime(cohort["t0"])

    print("[ehr] reading patients / labevents / diagnoses ...", flush=True)
    patients = pd.read_csv(args.patients, usecols=["subject_id", "anchor_age", "anchor_year"])
    anchor = patients.set_index("subject_id")[["anchor_age", "anchor_year"]].to_dict("index")

    keep_items = set(LAB_ITEMIDS.values())
    labevents = pd.read_csv(args.labevents, usecols=["subject_id", "itemid", "charttime", "valuenum"])
    labevents = labevents[labevents["itemid"].isin(keep_items) &
                          labevents["subject_id"].isin(set(cohort["subject_id"]))]
    labevents = labevents.dropna(subset=["valuenum"])
    labevents["charttime"] = pd.to_datetime(labevents["charttime"])
    labevents = labevents.sort_values(["subject_id", "itemid", "charttime"], kind="stable")
    print(f"[ehr] {len(labevents):,} lab rows for {cohort['subject_id'].nunique():,} cohort subjects "
          f"({time.time() - t_start:.0f} s)", flush=True)

    # per subject -> per itemid -> (sorted times ns, values)
    labs = {}
    t_all = labevents["charttime"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    v_all = labevents["valuenum"].to_numpy(dtype=float)
    s_all = labevents["subject_id"].to_numpy()
    i_all = labevents["itemid"].to_numpy()
    if len(s_all):
        brk = np.flatnonzero((s_all[1:] != s_all[:-1]) | (i_all[1:] != i_all[:-1])) + 1
        for a, b in zip(np.r_[0, brk], np.r_[brk, len(s_all)]):
            labs.setdefault(s_all[a], {})[i_all[a]] = (t_all[a:b], v_all[a:b])

    # comorbidities: earliest admission carrying each condition, per subject
    diagnoses = pd.read_csv(args.diagnoses, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    diagnoses = diagnoses[diagnoses["subject_id"].isin(set(cohort["subject_id"]))]
    admissions = pd.read_csv(args.admissions, usecols=["hadm_id", "admittime"])
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    dx = diagnoses.merge(admissions, on="hadm_id", how="left")
    dx["code"] = dx["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()
    first_dx = {}
    for cname, (p9, p10) in COMORBIDITY_PREFIXES.items():
        hit = (((dx["icd_version"] == 9) & dx["code"].str.startswith(p9)) |
               ((dx["icd_version"] == 10) & dx["code"].str.startswith(p10)))
        first_dx[cname] = dx[hit].groupby("subject_id")["admittime"].min().to_dict()
    print(f"[ehr] diagnoses indexed ({time.time() - t_start:.0f} s); extracting features ...", flush=True)

    lookback_ns = int(args.baseline_lookback_days) * _NS_PER_DAY
    empty = (np.array([], dtype=np.int64), np.array([], dtype=float))
    rows = []
    n = len(cohort)
    every = max(1, n // 100)
    sids = cohort["subject_id"].to_numpy()
    stids = cohort["study_id"].to_numpy()
    t0s = cohort["t0"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ages = cohort["age_at_t0"].to_numpy(dtype=float)
    genders = cohort["gender"].astype(str).str.upper().to_numpy() if "gender" in cohort else np.array([""] * n)
    t_loop = time.time()
    for k in range(n):
        sid, t0_ns = sids[k], t0s[k]
        lo_ns = t0_ns - lookback_ns
        female = genders[k] == "F"
        feat = {"subject_id": sid, "study_id": stids[k],
                "age_at_t0": float(ages[k]), "sex_female": 1.0 if female else 0.0}
        subj = labs.get(sid, {})
        for name, itemid in LAB_ITEMIDS.items():
            tt, vv = subj.get(itemid, empty)
            feat.update(_window_summary(tt, vv, t0_ns, lo_ns, name))

        # derived eGFR from in-window creatinine (physiologic range only)
        tt, vv = subj.get(CREATININE_ITEMID, empty)
        i0 = np.searchsorted(tt, lo_ns, "left"); i1 = np.searchsorted(tt, t0_ns, "left")
        ct, cv = tt[i0:i1], vv[i0:i1]
        ok = (cv > 0.1) & (cv < 30)
        ct, cv = ct[ok], cv[ok]
        if len(cv) and sid in anchor:
            years = ct.astype("datetime64[ns]").astype("datetime64[Y]").astype(int) + 1970
            age = real_age(anchor[sid]["anchor_age"], anchor[sid]["anchor_year"], years)
            eg = ckd_epi_2021(cv, np.asarray(age, dtype=float), female)
            feat.update(_window_summary(ct, eg, t0_ns, lo_ns, "egfr"))
        else:
            feat.update(_window_summary(empty[0], empty[1], t0_ns, lo_ns, "egfr"))

        t0 = pd.Timestamp(t0_ns)
        for cname in COMORBIDITY_PREFIXES:
            first = first_dx[cname].get(sid)
            feat[f"cmb_{cname}"] = 1.0 if (first is not None and pd.notna(first) and first < t0) else 0.0
        rows.append(feat)

        if (k + 1) % every == 0 or k + 1 == n:
            el = time.time() - t_loop
            eta = el / (k + 1) * (n - k - 1)
            print(f"\r[ehr] {k + 1:,}/{n:,} studies ({100 * (k + 1) / n:.0f}%)  "
                  f"elapsed {el / 60:.1f} min, ~{eta / 60:.1f} min left", end="", flush=True)
    print()

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("subject_id", "study_id")]
    out = os.path.join(args.out_dir, "ehr_features.parquet")
    df.to_parquet(out, index=False)
    with open(os.path.join(args.out_dir, "ehr_feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"[ehr] wrote {len(df)} rows x {len(feature_cols)} features -> {out} "
          f"({(time.time() - t_start) / 60:.1f} min total)")
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
