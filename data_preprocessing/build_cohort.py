"""Build the incident-CKD prognosis cohort from MIMIC-IV + MIMIC-IV-ECG.

This is the heart of turning a diagnostic ECG model into a *prognostic* one. The
design follows standard incident-disease cohort practice:

  baseline lookback        index time (t0)         prediction horizon
  |<------ features ------>|        | blanking |<------- label window ------->|
  patient history          ECG study   gap         CKD onset counted here

Key decisions (all configurable in configs/ckd_prognosis.yaml):

* **Index time (t0)**: the acquisition time of the patient's qualifying ECG
  study (the ECG that the model "sees"). Using the ECG time anchors the two
  modalities to the same moment.
* **Eligibility / exclusions** (must hold at/before t0):
    - no prior CKD diagnosis and no sustained low eGFR  -> incident, not prevalent
    - no ESRD / dialysis / transplant
    - baseline renal function known and >= 60 mL/min/1.73 m^2
    - at least ``min_baseline_days`` of prior history (so features are non-empty)
    - adult (age >= 18)
* **Blanking gap**: events within ``blank_days`` after t0 are dropped (avoid
  ambiguous near-baseline onset / reverse causation).
* **Label**: incident CKD onset (see ckd_labels.combined_ckd_onset) occurring in
  (t0 + blank_days, t0 + horizon_days]. For the survival head, the event time is
  binned into ``n_intervals``; subjects with no event are censored at last
  known follow-up (last lab / ECG / discharge).

Output: a parquet/csv "label table" with one row per eligible ECG study:
    subject_id, study_id, ecg_path, t0,
    label_binary, event_time_days, event_indicator, event_bin,
    split (filled later by make_splits.py)

NOTE: requires credentialed PhysioNet access to MIMIC-IV v3.x and
MIMIC-IV-ECG v1.0. Paths are configured via CLI args. Column names match the
public CSVs; adjust if your local extract differs.
"""

import argparse
import os

import numpy as np
import pandas as pd

from data_preprocessing.egfr import ckd_epi_2021
from data_preprocessing.ckd_labels import (
    first_ckd_diagnosis_time, sustained_low_egfr, combined_ckd_onset,
    is_esrd_or_rrt, is_ckd_diagnosis,
)

CREATININE_ITEMID = 50912  # labevents: Creatinine (blood chemistry), mg/dL


def _read(path, **kw):
    return pd.read_csv(path, **kw)


def real_age(anchor_age, anchor_year, event_year):
    """MIMIC-IV ages are anchored; recover age at the event year."""
    return anchor_age + (event_year - anchor_year)


def compute_egfr_table(labevents, patients):
    """Build subject/charttime/egfr from serum-creatinine lab events."""
    cr = labevents[labevents["itemid"] == CREATININE_ITEMID].copy()
    cr = cr.dropna(subset=["valuenum"])
    cr = cr[(cr["valuenum"] > 0.1) & (cr["valuenum"] < 30)]  # physiologic sanity
    cr["charttime"] = pd.to_datetime(cr["charttime"])

    p = patients[["subject_id", "gender", "anchor_age", "anchor_year"]]
    cr = cr.merge(p, on="subject_id", how="left")
    cr["age_at_draw"] = real_age(
        cr["anchor_age"], cr["anchor_year"], cr["charttime"].dt.year
    )
    cr["is_female"] = cr["gender"].str.upper().eq("F")
    cr["egfr"] = ckd_epi_2021(
        cr["valuenum"].to_numpy(),
        cr["age_at_draw"].to_numpy(),
        cr["is_female"].to_numpy(),
    )
    return cr[["subject_id", "charttime", "egfr", "valuenum", "age_at_draw", "is_female"]]


def baseline_egfr(egfr_table, t0_df, lookback_days):
    """Most recent eGFR in (t0 - lookback, t0] per (subject, study)."""
    e = egfr_table.merge(t0_df[["subject_id", "study_id", "t0"]], on="subject_id")
    e = e[(e["charttime"] <= e["t0"]) &
          (e["charttime"] >= e["t0"] - pd.to_timedelta(lookback_days, "D"))]
    e = e.sort_values("charttime").groupby(["subject_id", "study_id"]).tail(1)
    return e[["subject_id", "study_id", "egfr", "charttime"]].rename(
        columns={"egfr": "baseline_egfr", "charttime": "baseline_egfr_time"}
    )


def last_followup(egfr_table, ecg_records, admissions):
    """Last known contact per subject = max of last lab / last ECG / last discharge."""
    parts = []
    parts.append(egfr_table.groupby("subject_id")["charttime"].max())
    et = ecg_records.copy()
    et["ecg_time"] = pd.to_datetime(et["ecg_time"])
    parts.append(et.groupby("subject_id")["ecg_time"].max())
    ad = admissions.copy()
    ad["dischtime"] = pd.to_datetime(ad["dischtime"])
    parts.append(ad.groupby("subject_id")["dischtime"].max())
    fu = pd.concat(parts, axis=1)
    fu["last_followup"] = fu.max(axis=1)
    return fu[["last_followup"]].reset_index()


def build_cohort(args):
    os.makedirs(args.out_dir, exist_ok=True)

    patients = _read(args.patients)
    admissions = _read(args.admissions)
    diagnoses = _read(args.diagnoses)
    labevents = _read(
        args.labevents,
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
    )
    ecg = _read(args.ecg_record_list)  # subject_id, study_id, ecg_time, path/file_name

    # ---- 1. eGFR time series & CKD onset --------------------------------
    egfr_table = compute_egfr_table(labevents, patients)
    dx_onset = first_ckd_diagnosis_time(diagnoses, admissions)
    lab_onset = sustained_low_egfr(egfr_table)
    onset = combined_ckd_onset(dx_onset, lab_onset)

    # prevalent end-stage / RRT subjects to exclude entirely
    dx = diagnoses.copy()
    dx["is_esrd"] = dx.apply(lambda r: is_esrd_or_rrt(r["icd_code"], r["icd_version"]), axis=1)
    esrd_subjects = set(dx.loc[dx["is_esrd"], "subject_id"].unique())

    # ---- 2. candidate index times: each ECG study -----------------------
    ecg = ecg.rename(columns={c: c for c in ecg.columns})
    ecg["t0"] = pd.to_datetime(ecg["ecg_time"])
    path_col = "path" if "path" in ecg.columns else "file_name"
    t0_df = ecg[["subject_id", "study_id", "t0", path_col]].rename(
        columns={path_col: "ecg_path"}
    )

    # ---- 3. baseline renal function & age at t0 -------------------------
    base = baseline_egfr(egfr_table, t0_df, args.baseline_lookback_days)
    cohort = t0_df.merge(base, on=["subject_id", "study_id"], how="left")
    cohort = cohort.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id", how="left",
    )
    cohort["age_at_t0"] = real_age(
        cohort["anchor_age"], cohort["anchor_year"], cohort["t0"].dt.year
    )

    cohort = cohort.merge(onset[["subject_id", "ckd_onset_time"]], on="subject_id", how="left")
    fu = last_followup(egfr_table, ecg, admissions)
    cohort = cohort.merge(fu, on="subject_id", how="left")

    # ---- 4. eligibility / exclusions (incident, adult, baseline OK) -----
    n0 = len(cohort)
    cohort = cohort[cohort["age_at_t0"] >= 18]
    cohort = cohort[~cohort["subject_id"].isin(esrd_subjects)]
    cohort = cohort[cohort["baseline_egfr"].notna() & (cohort["baseline_egfr"] >= 60)]
    # exclude prevalent CKD: onset at or before t0 (+ small grace via blanking)
    prevalent = cohort["ckd_onset_time"].notna() & (
        cohort["ckd_onset_time"] <= cohort["t0"] + pd.to_timedelta(args.blank_days, "D")
    )
    cohort = cohort[~prevalent]
    # require minimum prior history
    have_hist = (cohort["t0"] - cohort["baseline_egfr_time"]).dt.days >= 0
    cohort = cohort[have_hist]
    print(f"[cohort] {n0} candidate studies -> {len(cohort)} eligible after exclusions")

    # ---- 5. labels, event time, censoring, binning ----------------------
    horizon = pd.to_timedelta(args.horizon_days, "D")
    t_event = cohort["ckd_onset_time"]
    t_censor = cohort["last_followup"].fillna(cohort["t0"])

    has_event = t_event.notna() & (t_event > cohort["t0"] + pd.to_timedelta(args.blank_days, "D"))
    # time-to-event (days from t0); censor at last follow-up if no event
    days_to_event = (t_event - cohort["t0"]).dt.days
    days_to_censor = (t_censor - cohort["t0"]).dt.days.clip(lower=0)

    event_indicator = has_event.astype(int)
    event_time_days = np.where(has_event, days_to_event, days_to_censor)
    event_time_days = np.maximum(event_time_days, 0)

    # fixed-horizon binary label (eligible = event-by-horizon OR followed past horizon)
    label_binary = ((event_indicator == 1) & (event_time_days <= args.horizon_days)).astype(int)
    followed_to_horizon = event_time_days >= args.horizon_days
    eligible_binary = (label_binary == 1) | followed_to_horizon

    # discrete-time bins over [0, horizon]
    edges = np.linspace(0, args.horizon_days, args.n_intervals + 1)
    capped = np.minimum(event_time_days, args.horizon_days - 1e-6)
    event_bin = np.clip(np.digitize(capped, edges) - 1, 0, args.n_intervals - 1)

    cohort = cohort.assign(
        label_binary=label_binary,
        eligible_binary=eligible_binary.astype(int),
        event_time_days=event_time_days,
        event_indicator=event_indicator,
        event_bin=event_bin,
    )

    keep = [
        "subject_id", "study_id", "ecg_path", "t0", "age_at_t0", "gender",
        "baseline_egfr", "label_binary", "eligible_binary",
        "event_time_days", "event_indicator", "event_bin",
    ]
    cohort = cohort[keep].reset_index(drop=True)

    out_path = os.path.join(args.out_dir, "ckd_cohort_labels.parquet")
    cohort.to_parquet(out_path, index=False)
    print(f"[cohort] wrote {len(cohort)} rows -> {out_path}")
    print(cohort["label_binary"].value_counts(normalize=True).rename("label rate"))
    return cohort


def get_parser():
    p = argparse.ArgumentParser(description="Build incident-CKD prognosis cohort")
    p.add_argument("--patients", required=True, help="hosp/patients.csv[.gz]")
    p.add_argument("--admissions", required=True, help="hosp/admissions.csv[.gz]")
    p.add_argument("--diagnoses", required=True, help="hosp/diagnoses_icd.csv[.gz]")
    p.add_argument("--labevents", required=True, help="hosp/labevents.csv[.gz]")
    p.add_argument("--ecg_record_list", required=True, help="mimic-iv-ecg/record_list.csv")
    p.add_argument("--out_dir", default="dataset/ckd")
    p.add_argument("--horizon_days", type=int, default=730, help="prediction horizon (default 2y)")
    p.add_argument("--blank_days", type=int, default=30, help="blanking gap after t0")
    p.add_argument("--baseline_lookback_days", type=int, default=365)
    p.add_argument("--n_intervals", type=int, default=8, help="survival time bins")
    return p


if __name__ == "__main__":
    build_cohort(get_parser().parse_args())
