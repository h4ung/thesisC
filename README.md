# Multimodal Risk Prediction for Incident Chronic Kidney Disease

Multimodal **prognostic risk-prediction** model that estimates each patient's
**risk of developing incident chronic kidney disease (CKD)** over a future
horizon (1 and 2 years), and stratifies patients into **low- and high-risk
groups**, by combining the **MIMIC-IV** electronic health record (EHR) with the
**MIMIC-IV-ECG** 12-lead waveform dataset.

## Quick start

```bash
# 0. check the whole pipeline without MIMIC (synthetic data, results meaningless)
python scripts/make_synthetic_dataset.py --out_dir dataset/synthetic
python run.py --config configs/ckd_prognosis.yaml --root_path dataset/synthetic \
  --setting synthetic_check --train_epochs 2 --use_gpu 0 --num_workers 0

# 1. real data
bash scripts/preprocess_all.sh                     # MIMIC-IV + MIMIC-IV-ECG -> dataset/ckd
python run.py --config configs/ckd_prognosis.yaml  # train + test the reference model
python scripts/plot_results.py --run results/ckd_prognosis/ckd_2y_survival_multimodal

# 2. ablation study
python run_ablations.py --plan configs/ablations.yaml
python collect_results.py --plan configs/ablations.yaml
```

## What the model outputs (per patient)

| output | meaning |
|---|---|
| `risk_score` | continuous ranking score, higher = CKD expected sooner |
| `abs_risk_365d`, `abs_risk_730d` | predicted probability of incident CKD within 1 y / 2 y |
| `risk_group` | `low` / `high` (or `low` / `intermediate` / `high`), cut-points fitted on validation |

Written to `results/<...>/<setting>/test_predictions.csv`, with
`risk_groups.json` (observed Kaplan-Meier risk per group, log-rank p, HR high vs
low), `km_curves.csv`, `calibration.csv` and `test_metrics.json`.


## Reported metrics

| metric | computed over |
|---|---|
| `c_index` | all patients, follow-up truncated at the horizon (Harrell); `c_index_untruncated` for comparison with Thesis B |
| `td_auroc_<h>d` | **eligible patients only** — event by h, or follow-up reaching h |
| `brier_ipcw_<h>d` | all patients, inverse-probability-of-censoring weighted (Graf et al.) |
| `ece_730d` | expected calibration error over risk deciles (KM-observed vs predicted) |
| `*_ci95` | subject-level bootstrap 95% CI (`--n_bootstrap`) |
| `hr_high_vs_low`, `logrank_p` | separation of the risk groups on the test set |

Patients censored *before* the horizon have an unknown horizon label; scoring
them as negatives inflates `td_auroc` (see `tests/test_metrics.py`).
