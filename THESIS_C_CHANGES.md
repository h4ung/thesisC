# Thesis C changes

Supervisor feedback at the start of Thesis C:

1. update the prognostic layer to do **risk prediction, not binary classification**;
2. **identify high-risk and low-risk patients**;
3. implement **ablation studies**.

This file records what changed in the code for each point, plus one labelling
bug found along the way. Test status: **58 tests pass** (`pytest tests -q`),
including an end-to-end train → test → reload run on synthetic data and
cross-checks of the survival statistics against `lifelines`.

---

## 0. Bug fix: CKD onsets after the horizon were counted as in-horizon events

**Files:** `data_provider/data_loader.py`, `exp/exp_ckd_prognosis.py`, `utils/survival.py`

`build_cohort.py` sets `event_indicator = 1` for a CKD onset at *any* time after
t0 and caps `event_bin` at the last interval. The survival loss therefore
treated an onset at year 5 as an event in the 1.75–2 y bin, inflating the
predicted late hazard. The dataset now applies **administrative censoring at the
horizon** (`event_indicator_h`, `event_time_h`); every risk head trains on these.
No cohort rebuild is needed. The C-index is now computed on horizon-truncated
follow-up (`c_index`); the previous definition is kept as `c_index_untruncated`.

**Action required:** survival-head results from Thesis B should be re-run.

## 1. Prognostic layer → risk prediction

**Files:** `models/heads.py`, `models/CardioformerCKD.py`, `utils/losses.py`, `exp/exp_ckd_prognosis.py`, `run.py`

* Every head now produces a **continuous risk score** and an **absolute risk**
  (probability of incident CKD) at each horizon in `--eval_horizons_days`
  (default 1 y and 2 y).
* `--head survival` (default): discrete-time hazards → cumulative incidence curve.
* `--head cox` (new): DeepSurv-style log-risk trained with the Cox partial
  likelihood (Breslow ties, verified against a hand-computed likelihood and
  against lifelines). Absolute risk comes from a Breslow baseline hazard fitted on
  the training split after training (`baseline_hazard.json`), verified against
  lifelines to 1e-16.
* `--head binary` is kept **only as an ablation baseline**.
* `--head_hidden 0` makes the head linear; `--ehr_mode identity` feeds raw
  standardised EHR features. Together with `--use_ecg 0 --head cox` this is a
  classical **linear Cox PH baseline** trained on identical splits.
* New evaluation: time-dependent AUROC per horizon, **IPCW Brier score**
  (Graf et al.), **calibration by deciles + ECE**, and **subject-level bootstrap
  95% CIs** (`--n_bootstrap`).

## 2. High-risk / low-risk patients

**Files:** `utils/risk_stratification.py`, `utils/survival.py`, `exp/exp_ckd_prognosis.py`, `scripts/plot_results.py`

* Cut-points are fitted on the **validation** split only, then applied to test.
* Methods: `quantile` (default median split → low/high; `0.33,0.67` → tertiles),
  `absolute` (fixed predicted-risk cut-offs), `youden`, `sensitivity`.
* Per group: n, observed Kaplan–Meier CKD incidence at the horizon with 95% CI,
  mean predicted risk. Between groups: log-rank test and hazard ratio (high vs low).
* Outputs: `test_predictions.csv` (per-patient risk and group),
  `risk_groups.json`, `km_curves.csv`, `calibration.csv`; figures from
  `scripts/plot_results.py` (KM curves by risk group, calibration plot, risk
  distribution).
* Re-stratify a trained model without retraining: `--is_training 0 --risk_strat_...`.

## 3. Ablation studies

**Files:** `configs/ablations.yaml`, `run_ablations.py`, `collect_results.py`, `layers/Embed.py`, `layers/Cardioformer_EncDec.py`, `models/Cardioformer.py`, `data_provider/data_loader.py`

One-factor-at-a-time variants vs the reference model, each over 3 seeds:

| group | variants |
|---|---|
| modality | ECG+EHR, ECG only, EHR only, linear Cox (EHR) |
| fusion | gated, concat, cross-attention over patches, over granularities |
| prognostic layer | survival, Cox, binary |
| ECG backbone | single granularity, no inter-granularity attention, channel-independent patching, linear patch embedding |
| EHR features | − renal labs, − other labs, − comorbidities, ECG + demographics, ECG only |
| training | no augmentation |

New switches: `--patch_encoder resnet|linear`, `--inter_granularity 0/1`,
`--ehr_drop_groups`. Defaults reproduce the Thesis B architecture exactly (state-dict keys
unchanged, so old checkpoints still load).

`collect_results.py` writes mean ± std over seeds and the **paired Δ vs the reference**
(seed by seed) as CSV, Markdown and LaTeX tables. `scripts/plot_results.py --ablation`
draws the Δ C-index chart.

## 4. Tooling

* `scripts/make_synthetic_dataset.py`: small synthetic dataset with the real schema,
  so the whole pipeline can be checked without MIMIC access. **Never report its numbers.**
* `tests/test_risk_layer.py`, `tests/test_ablation_and_e2e.py`: 24 new tests.
* Notebook sections 7–10 updated for stratification, ablations and per-patient risk.

## 5. pandas 3 compatibility (CI fix)

CI's Python 3.11 job installs pandas 3, whose copy-on-write makes `.to_numpy()`
return read-only arrays; shuffling them in place raised `ValueError: array is
read-only` in `data_preprocessing/make_splits.py` (and the synthetic generator).
Both now shuffle a copy. `tests/test_ablation_and_e2e.py::test_make_splits_is_subject_independent_under_pandas3`
guards it. Verified under pandas 2.3 and 3.0.

## Behavioural changes

| change | effect |
|---|---|
| horizon censoring | survival-head numbers from Thesis B are not comparable; re-run |
| `c_index` now horizon-truncated | use `c_index_untruncated` to compare with Thesis B logs |
| `brier_at_horizon` now IPCW-weighted over all patients | differs from Thesis B's unweighted eligible-only value |
| early stopping on `--early_stop_metric` (default C-index) for every head | binary head previously early-stopped on AUROC |
| `test()` writes predictions / groups / curves | extra files in each results dir |

## 6. Preprocessing speed and cohort reporting (6 Oct)

* `extract_ehr_features.py` rewritten for speed: the old loop rebuilt the
  patients index twice per ECG study and ran ~20 pandas filters per study
  (many hours on the full cohort, no progress output). Labs are now sorted
  once per subject and windowed by binary search; a progress line with an
  ETA is printed. Output is identical to the old version
  (`tests/test_ehr_features_fast.py` compares them value by value).
* `build_cohort.py` now records the ECG-study and subject counts after each
  exclusion step in `dataset/ckd/cohort_flow.csv` (Table 4.1). Final cohort
  is unchanged.
* `scripts/cohort_table.py` prints Table 4.1 and the Table 4.2 numbers.
* `scripts/filter_labevents.py` shrinks labevents to the 8 lab tests used.
