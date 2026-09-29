"""CKD case definitions.

Two complementary signals define chronic kidney disease:

1. **Diagnosis codes** — ICD-9 ``585.x`` and ICD-10 ``N18.x`` (chronic kidney
   disease), plus ESRD / dialysis / transplant codes used for the exclusion of
   prevalent end-stage disease.
2. **Laboratory (eGFR) criterion** — KDIGO: eGFR < 60 mL/min/1.73 m^2 sustained
   for >= 90 days (i.e. confirmed by a second qualifying value at least 90 days
   later). The "sustained" rule is what separates *chronic* from *acute* kidney
   injury and is essential for label quality.

The functions here are pure (operate on pandas frames) so they can be unit
tested without database access.
"""

import numpy as np
import pandas as pd

# ICD prefixes ---------------------------------------------------------------
CKD_ICD9_PREFIXES = ("585",)            # 585.x chronic kidney disease (585.6 = ESRD)
CKD_ICD10_PREFIXES = ("N18",)           # N18.x chronic kidney disease (N18.6 = ESRD)

# Prevalent end-stage / RRT that should also be excluded at baseline
ESRD_ICD9 = ("5856", "V451", "V560", "V568")           # ESRD, dialysis status
ESRD_ICD10 = ("N186", "Z992", "Z9115", "Z49", "T861")  # ESRD, dialysis, transplant

SUSTAINED_DAYS = 90
EGFR_THRESHOLD = 60.0


def _norm(code):
    return str(code).replace(".", "").upper().strip()


def is_ckd_diagnosis(icd_code, icd_version):
    code = _norm(icd_code)
    if int(icd_version) == 9:
        return any(code.startswith(p) for p in CKD_ICD9_PREFIXES)
    return any(code.startswith(p) for p in CKD_ICD10_PREFIXES)


def is_esrd_or_rrt(icd_code, icd_version):
    code = _norm(icd_code)
    table = ESRD_ICD9 if int(icd_version) == 9 else ESRD_ICD10
    return any(code.startswith(p) for p in table)


def first_ckd_diagnosis_time(dx_df, admissions_df):
    """Earliest CKD diagnosis time per subject.

    dx_df         : diagnoses_icd (subject_id, hadm_id, icd_code, icd_version)
    admissions_df : admissions (hadm_id, admittime) to time-stamp the diagnosis.
    Returns DataFrame[subject_id, ckd_dx_time].
    """
    dx = dx_df.copy()
    dx["is_ckd"] = dx.apply(
        lambda r: is_ckd_diagnosis(r["icd_code"], r["icd_version"]), axis=1
    )
    dx = dx[dx["is_ckd"]]
    dx = dx.merge(admissions_df[["hadm_id", "admittime"]], on="hadm_id", how="left")
    dx["admittime"] = pd.to_datetime(dx["admittime"])
    out = dx.groupby("subject_id")["admittime"].min().reset_index()
    return out.rename(columns={"admittime": "ckd_dx_time"})


def sustained_low_egfr(egfr_df, threshold=EGFR_THRESHOLD, sustained_days=SUSTAINED_DAYS):
    """First time a subject has a *sustained* low eGFR (chronic, not acute AKI).

    egfr_df : DataFrame[subject_id, charttime, egfr] (one row per creatinine draw)
    Returns DataFrame[subject_id, ckd_lab_time] = time of the FIRST low value that
    is later confirmed by another low value >= ``sustained_days`` afterwards.
    """
    df = egfr_df.copy()
    df["charttime"] = pd.to_datetime(df["charttime"])
    df = df.sort_values(["subject_id", "charttime"])
    results = []
    for sid, g in df.groupby("subject_id"):
        low = g[g["egfr"] < threshold]
        if low.empty:
            continue
        times = low["charttime"].to_numpy()
        found = None
        for i, t0 in enumerate(times):
            # any later low value at least `sustained_days` after t0 confirms it
            horizon = t0 + np.timedelta64(sustained_days, "D")
            if (times[i + 1:] >= horizon).any():
                found = t0
                break
        if found is not None:
            results.append((sid, found))
    return pd.DataFrame(results, columns=["subject_id", "ckd_lab_time"])


def combined_ckd_onset(dx_onset, lab_onset):
    """Earliest of diagnosis-based and lab-based onset per subject."""
    merged = pd.merge(dx_onset, lab_onset, on="subject_id", how="outer")
    merged["ckd_onset_time"] = merged[["ckd_dx_time", "ckd_lab_time"]].min(axis=1)
    return merged[["subject_id", "ckd_onset_time", "ckd_dx_time", "ckd_lab_time"]]
