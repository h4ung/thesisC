#!/usr/bin/env bash
# End-to-end preprocessing: MIMIC-IV + MIMIC-IV-ECG -> model-ready dataset.
# Requires credentialed PhysioNet downloads. Edit the paths below.
set -euo pipefail

# ---- paths to the raw data (EDIT THESE) ------------------------------------
MIMIC=../mimic-iv-3.1                      # contains hosp/
ECG=../mimic-iv-ecg          # contains record_list.csv + files/
OUT=dataset/ckd

HOSP=$MIMIC/hosp

# ---- 1. cohort + labels ----------------------------------------------------
python -m data_preprocessing.build_cohort \
  --patients   $HOSP/patients.csv \
  --admissions $HOSP/admissions.csv \
  --diagnoses  $HOSP/diagnoses_icd.csv \
  --labevents  $HOSP/labevents.csv \
  --ecg_record_list $ECG/record_list.csv \
  --out_dir $OUT \
  --horizon_days 730 --blank_days 30 --baseline_lookback_days 365 --n_intervals 8

# ---- 2. structured EHR features --------------------------------------------
python -m data_preprocessing.extract_ehr_features \
  --cohort $OUT/ckd_cohort_labels.parquet \
  --patients   $HOSP/patients.csv \
  --admissions $HOSP/admissions.csv \
  --diagnoses  $HOSP/diagnoses_icd.csv \
  --labevents  $HOSP/labevents.csv \
  --out_dir $OUT

# ---- 3. ECG waveforms ------------------------------------------------------
python -m data_preprocessing.extract_ecg \
  --cohort $OUT/ckd_cohort_labels.parquet \
  --ecg_root $ECG \
  --out_dir $OUT/ecg_npy --target_fs 250

# ---- 4. link + split -------------------------------------------------------
python -m data_preprocessing.link_ecg_ehr \
  --cohort $OUT/ckd_cohort_labels.parquet \
  --ehr $OUT/ehr_features.parquet \
  --ecg_manifest $OUT/ecg_npy/ecg_manifest.parquet \
  --out_dir $OUT --policy all

python -m data_preprocessing.make_splits \
  --dataset $OUT/ckd_dataset.parquet --out_dir $OUT \
  --train_frac 0.6 --val_frac 0.2 --seed 41

echo "Done. Train with: python run.py --config configs/ckd_prognosis.yaml"
