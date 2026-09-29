#!/usr/bin/env bash
# Train CardioformerCKD for 2-year incident-CKD RISK prediction (multimodal, survival risk layer).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python run.py \
  --task_name ckd_prognosis --is_training 1 \
  --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_survival_multimodal \
  --use_ecg 1 --use_ehr 1 \
  --head survival --n_intervals 8 --horizon_days 730 --fusion gated \
  --batch_size 16 --learning_rate 1e-4 --train_epochs 30 --patience 6
