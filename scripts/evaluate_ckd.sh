#!/usr/bin/env bash
# Evaluate the trained reference model and run the full ablation study (Thesis C).
#
# --is_training 0 loads <checkpoints>/<setting>/checkpoint.pth (and, for the Cox
# head, baseline_hazard.json) and fails loudly if it is missing.
#
# CLI flags override the YAML (run.py::resolve_args); run_ablations.py relies on
# this so every variant is a real ablation with its own checkpoint/results dir.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 1. evaluate the trained reference model: risk metrics + low/high risk groups
python run.py --is_training 0 --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_survival_multimodal

# 1b. same model, alternative stratifications (no retraining needed)
python run.py --is_training 0 --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_survival_multimodal --results results/ckd_prognosis/tertiles \
  --risk_strat_quantiles 0.33,0.67
python run.py --is_training 0 --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_survival_multimodal --results results/ckd_prognosis/absolute \
  --risk_strat_method absolute --risk_strat_cutoffs 0.05,0.15

python scripts/plot_results.py --run results/ckd_prognosis/ckd_2y_survival_multimodal

# 2. ablation study (all groups x 3 seeds; see configs/ablations.yaml)
python run_ablations.py --plan configs/ablations.yaml
python collect_results.py --plan configs/ablations.yaml
python scripts/plot_results.py --ablation results/ckd_prognosis/ablation_summary.csv
