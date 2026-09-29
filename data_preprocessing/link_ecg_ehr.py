"""Resolve which ECG study defines each index time and merge all tables.

``build_cohort.py`` already treats every ECG study as a candidate index. This
step lets you optionally collapse to ONE qualifying ECG per subject (e.g. the
earliest ECG that satisfies eligibility) to avoid many correlated studies from
the same patient leaking across splits, and produces the final joined table the
dataloader consumes:

    subject_id, study_id, ecg_npy, t0, <labels>, <ehr feature columns>

Selection policies (``--policy``):
  * ``all``      keep every eligible study (subject-independent split still
                 guarantees a subject appears in only one split).
  * ``earliest`` one row per subject: earliest eligible ECG (most prospective).
  * ``latest``   one row per subject: latest eligible ECG before any event.
"""

import argparse
import os

import pandas as pd


def run(args):
    cohort = pd.read_parquet(args.cohort)
    ehr = pd.read_parquet(args.ehr)
    manifest = pd.read_parquet(args.ecg_manifest)

    df = cohort.merge(manifest[["study_id", "npy"]], on="study_id", how="inner")
    df = df.rename(columns={"npy": "ecg_npy"})

    # The EHR feature table is the source of truth for any feature it also
    # computes (e.g. age_at_t0). Drop overlapping non-key columns from the cohort
    # side before merging so pandas does not create _x / _y suffixes.
    keys = ["subject_id", "study_id"]
    overlap = [c for c in ehr.columns if c in df.columns and c not in keys]
    df = df.drop(columns=overlap)
    df = df.merge(ehr, on=keys, how="left")

    if args.policy != "all":
        df["t0"] = pd.to_datetime(df["t0"])
        df = df.sort_values("t0")
        keep = "first" if args.policy == "earliest" else "last"
        df = df.drop_duplicates(subset="subject_id", keep=keep)

    out = os.path.join(args.out_dir, "ckd_dataset.parquet")
    df.reset_index(drop=True).to_parquet(out, index=False)
    print(f"[link] {len(df)} rows (policy={args.policy}) -> {out}")


def get_parser():
    p = argparse.ArgumentParser(description="Link ECG + EHR + labels")
    p.add_argument("--cohort", default="dataset/ckd/ckd_cohort_labels.parquet")
    p.add_argument("--ehr", default="dataset/ckd/ehr_features.parquet")
    p.add_argument("--ecg_manifest", default="dataset/ckd/ecg_npy/ecg_manifest.parquet")
    p.add_argument("--out_dir", default="dataset/ckd")
    p.add_argument("--policy", choices=["all", "earliest", "latest"], default="all")
    return p


if __name__ == "__main__":
    run(get_parser().parse_args())
