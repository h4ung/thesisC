"""Print Table 4.1 (cohort flow) and the numbers for Table 4.2 / Section 8.1.

Run after preprocessing (step 6d), from the repo root:
    python scripts/cohort_table.py            # uses dataset/ckd
    python scripts/cohort_table.py --root_path dataset/ckd

Reads dataset/ckd/cohort_flow.csv (written by build_cohort.py) and
dataset/ckd/ckd_dataset_split.parquet (written by make_splits.py).
"""

import argparse
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", default="dataset/ckd")
    p.add_argument("--horizon_days", type=float, default=730)
    a = p.parse_args()

    flow_path = os.path.join(a.root_path, "cohort_flow.csv")
    if not os.path.exists(flow_path):
        raise SystemExit(f"{flow_path} not found: re-run step 6a with the updated build_cohort.py")
    flow = pd.read_csv(flow_path)

    split_path = os.path.join(a.root_path, "ckd_dataset_split.parquet")
    df = pd.read_parquet(split_path) if os.path.exists(split_path) else None
    if df is not None:
        flow = pd.concat([flow, pd.DataFrame([{
            "step": "Linked to preprocessed ECG and EHR features",
            "ecg_studies": len(df), "subjects": df["subject_id"].nunique(),
            "removed_studies": int(flow["ecg_studies"].iloc[-1] - len(df))}])], ignore_index=True)

    print("\nTable 4.1 - cohort flow")
    print(flow[["step", "ecg_studies", "subjects"]].to_string(index=False))

    if df is None:
        print("\n(Run steps 6b-6d to get the final linked row and the split summary.)")
        return

    H = a.horizon_days
    t, e = df["event_time_days"], df["event_indicator"]
    df = df.assign(event_by_h=((e == 1) & (t <= H)).astype(int),
                   censored_early=((e == 0) & (t < H)).astype(int))
    known = df[df["eligible_binary"] == 1]
    print("\nTable 4.2 / Section 8.1 - by split")
    summ = df.groupby("split").agg(
        ecg_studies=("study_id", "size"), subjects=("subject_id", "nunique"),
        age_mean=("age_at_t0", "mean"), female_pct=("sex_female", "mean"),
        baseline_egfr_median=("egfr_last", "median") if "egfr_last" in df else ("study_id", "size"),
        ckd_events_2y=("event_by_h", "sum"),
        censored_before_2y_pct=("censored_early", "mean"),
        median_followup_days=("event_time_days", "median"))
    summ["female_pct"] = (100 * summ["female_pct"]).round(1)
    summ["censored_before_2y_pct"] = (100 * summ["censored_before_2y_pct"]).round(1)
    summ["age_mean"] = summ["age_mean"].round(1)
    summ = summ.reindex(["train", "val", "test"])
    print(summ.to_string())
    for c in ("cmb_diabetes", "cmb_hypertension", "cmb_heart_failure"):
        if c in df:
            print(f"  {c}: " + ", ".join(f"{s} {100 * g[c].mean():.1f}%" for s, g in df.groupby("split")))
    print(f"\n2-year CKD rate among patients with known 2-year status: "
          f"{known['label_binary'].mean():.1%} ({int(known['label_binary'].sum())} / {len(known)})")
    print(f"Share censored before 2 years (what a binary label would discard): "
          f"{df['censored_early'].mean():.1%}")


if __name__ == "__main__":
    main()
