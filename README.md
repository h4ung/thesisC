# Cardioformer-CKD: Multimodal Risk Prediction for Incident Chronic Kidney Disease

Multimodal **prognostic risk-prediction** model that estimates each patient's
**risk of developing incident chronic kidney disease (CKD)** over a future
horizon (1 and 2 years), and stratifies patients into **low- and high-risk
groups**, by combining the **MIMIC-IV** electronic health record (EHR) with the
**MIMIC-IV-ECG** 12-lead waveform dataset.

> **Thesis C update.** The prognostic layer now does *risk prediction*, not
> binary classification; risk stratification and a full ablation study were
> added. See **`THESIS_C_CHANGES.md`** for what changed and why, and
> `PATCHES.md` for the Thesis B correctness fixes.

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

## Prognostic risk layer (`--head`)

| head | trained with | risk curve | role |
|---|---|---|---|
| `survival` (default) | discrete-time survival NLL | CIF at every bin | main model |
| `cox` | Cox partial likelihood | Breslow baseline x exp(eta) | alternative risk layer |
| `binary` | focal BCE on horizon-eligible patients | horizon only | **ablation baseline only** |

## Risk stratification (`--risk_strat_method`)

`quantile` (default `--risk_strat_quantiles 0.5`: median split into low / high;
`0.33,0.67` for tertiles), `absolute` (`--risk_strat_cutoffs 0.05,0.15` on
predicted 2-year risk), `youden`, or `sensitivity` (`--risk_strat_target_sens`).
Cut-points are **always fitted on the validation split** and frozen before the
test split is touched.

## Ablation study

`configs/ablations.yaml` defines one-factor-at-a-time variants relative to the
reference model, grouped as: modality, fusion, prognostic layer, ECG backbone
components (multi-granularity, inter-granularity attention, cross-channel
patching, ResNet patch encoder), EHR feature groups, training, and a classical
linear Cox baseline. Each variant runs for 3 seeds; `collect_results.py` writes
mean ± std and the paired Δ vs the reference as CSV / Markdown / LaTeX.

## Provenance of the ECG backbone (important for the thesis)

`models/Cardioformer.py`, `layers/Embed.py` and `layers/Cardioformer_EncDec.py`
are an **independent clean-room re-implementation** of the Cardioformer encoder,
written from the description in the paper (Mobin et al.,
*Cardioformer: Advancing AI in ECG Analysis with Multi-Granularity Patching and
ResNet*, [arXiv:2505.05538](https://arxiv.org/abs/2505.05538), 2025). **No code
from the authors' repository was used, and this implementation has not been
validated against their released weights or results.**

Therefore:

* published Cardioformer numbers are **not** directly comparable to numbers
  produced here — the ECG-only baseline should be reported as *"our
  re-implementation of Cardioformer"*, never as *"Cardioformer"*;
* details the paper leaves unspecified (patch strides, router-token design,
  normalisation placement, initialisation) are our own choices, documented in the
  module docstrings;
* a gap against the published results is expected and is a property of the
  re-implementation, not a failed reproduction.

## Configuration precedence

```
explicit CLI flag  >  --config YAML  >  argparse default
```

`run.py::resolve_args` only writes a YAML key into the namespace if that flag was
**not** typed on the command line. This is what makes

```bash
python run.py --config configs/ckd_prognosis.yaml --setting ckd_2y_ecg_only --use_ehr 0
```

a real ablation (and keeps it writing to its own checkpoint/results directory
rather than the `setting` declared inside the YAML).

## Evaluating a trained model

```bash
python run.py --is_training 0 --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_survival_multimodal          # loads checkpoints/<setting>/checkpoint.pth
python run.py --is_training 0 --checkpoint_path path/to/checkpoint.pth ...   # or point at one
```

`test()` always loads the checkpoint from disk before scoring. A missing
checkpoint raises `FileNotFoundError` rather than silently evaluating randomly
initialised weights.

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

See `PATCHES.md` for the full list of fixes and their effect on results.
