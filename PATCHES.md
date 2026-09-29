# Patches — correctness fixes

Seven issues, in order of how badly they affect reported numbers. The first three
would have **silently invalidated results**: nothing crashes, the metrics just
come out wrong.

---

## 1. YAML silently overrode the CLI — the ablations did not ablate

**Files:** `run.py`

`apply_yaml()` ran *after* `parse_args()` and unconditionally `setattr`-ed every
YAML key onto the namespace, so YAML beat the CLI (the opposite of the
docstring). Consequences:

* `--config configs/ckd_prognosis.yaml --use_ehr 0` ran with `use_ehr=True`. The
  "ECG-only" and "EHR-only" ablations were **both the full multimodal model**.
* `--setting ckd_2y_ecg_only` was overwritten by `setting: ckd_2y_survival_multimodal`
  from the YAML, so every ablation wrote to the *same* checkpoint and results
  directory, overwriting the previous run.

**Fix:** `resolve_args()` establishes `explicit CLI > YAML > argparse default`.
A second parser with `default=argparse.SUPPRESS` reveals exactly which flags were
typed; YAML only fills the rest. YAML values are cast through the same `type=`
callable argparse would use, unknown YAML keys warn, and a missing `--config`
raises instead of being ignored. `run.py` no longer imports torch at module level,
so the precedence logic is unit-testable on a bare box.

**Tests:** `tests/test_config_override.py`
**Action required:** any ablation table produced before this fix must be re-run.

---

## 2. `test()` never loaded a checkpoint

**Files:** `exp/exp_basic.py`, `exp/exp_ckd_prognosis.py`

Only `train()` restored weights, at the very end of the loop. `--is_training 0`
therefore built a fresh model and evaluated **randomly initialised weights** —
producing a plausible-looking C-index near 0.5 and no error.

**Fix:** `Exp_Basic.load_checkpoint()` / `checkpoint_file()`; `test()` always
loads from disk before scoring, and raises `FileNotFoundError` when the checkpoint
is absent. Added `--checkpoint_path` to point at an arbitrary `.pth`. Eval-only
runs also reload the `ehr_scaler.json` saved next to the checkpoint, so test-time
preprocessing is identical to training. `torch.load` is called with
`weights_only=True` (with a fallback for older torch).

**Tests:** `tests/test_checkpoint_and_schedule.py`

---

## 3. Time-dependent AUROC counted early-censored patients as negatives

**Files:** `exp/exp_ckd_prognosis.py`, `utils/metrics.py`

`time_dependent_auroc` accepts an `eligible_mask`, but `validate()` never passed
one. A patient censored at day 50 has an **unknown** 2-year label; scoring them as
a negative pads the negative class with easy cases the model was never tested on
and inflates the metric.

**Fix:** `validate()` collects `eligible_binary` (defined in
`data_preprocessing/build_cohort.py` as *event by horizon* OR *followed to
horizon*) and passes it as the mask for all fixed-horizon metrics. Metrics now
report `n_eligible` / `n_total` so the drop is visible. Added
`utils.metrics.horizon_eligibility()` as a fallback, plus `brier_at_horizon`.
The C-index still uses everyone — censoring is handled by the comparable-pair
definition there.

**Tests:** `tests/test_metrics.py::test_td_auroc_is_inflated_when_censored_counted_as_negative`
(a worked example where the honest AUROC is 0.50 and the buggy one reads 0.80).

---

## 4. `lradj: type1` halved the LR every epoch

**Files:** `utils/tools.py`, `run.py`, `configs/*.yaml`

`0.5 ** (epoch - 1)` takes `1e-4` to `~2e-7` by epoch 10 and `~1e-13` by epoch 30
(the configured budget). Training stopped learning long before early stopping
triggered, and the "converged" model was whatever epoch 3–4 produced.

**Fix:** default is now `cosine` over `train_epochs`. Added `step`
(`--lr_decay_every`, `--lr_decay_rate`) and `none`. `type1` is retained, floored
at `--min_lr`, and documented as legacy-only. The schedule is now stepped with
`epoch + 2`, i.e. it sets the LR for the *next* epoch, so epoch 1 trains at the
full configured LR. The current LR is printed each epoch.

**Tests:** `tests/test_checkpoint_and_schedule.py`

---

## 5. C-index was an O(n²) Python double loop

**Files:** `utils/metrics.py`

Fine for a validation split, minutes-long on a full test cohort.

**Fix:** uses `lifelines.utils.concordance_index` when installed (note the sign
flip — lifelines expects a score where higher = *longer* survival, we predict
risk), otherwise a blocked, vectorised numpy path. Same pair definition and tie
handling, streamed in blocks so no `n × n` matrix is allocated. Verified
bit-identical to the original loop on random cohorts with ties in both time and
risk; 20,000 patients in ~0.8 s.

**Tests:** `tests/test_metrics.py` (compares against the original loop, kept in
the test file as ground truth).

---

## 6. `cross` fusion attended over G tokens, not the patch sequence

**Files:** `models/CardioformerCKD.py`, `models/Cardioformer.py`,
`layers/Cardioformer_EncDec.py`

`encode()` mean-pooled each granularity before returning, and `_encode_ecg()` then
reshaped that pooled vector into `(B, G, d_model)`. So the "cross-attention"
had only **G = 9–12 keys** — one summary per granularity — and could not attend to
the waveform at all. It was closer to a gated pool over granularities than to the
cross-modal attention the docstring describes.

**Fix:** `CardioformerEncoder.forward(..., return_streams=True)` and
`Model.encode(..., return_tokens=True)` expose the full multi-granularity token
sequence `(B, T, d_model)`, `T = Σ_g ⌈seq_len / patch_len_g⌉` (≈ 350 tokens for
`seq_len=250` with the 12-granularity list). `CrossAttentionFusion` now attends
over those, with a residual + LayerNorm around the EHR query, and stashes the
attention weights on `.last_attn` for interpretability plots — which is a genuinely
useful thesis figure: *which part of the ECG, at which scale, does this patient's
EHR profile make the model look at?*

The old behaviour is kept as `--cross_attn_tokens granularity`, so
patches-vs-granularity is now a reportable ablation (`scripts/evaluate_ckd.sh`).

**Tests:** `tests/test_model_forward.py` (asserts the attention map is
`(B, 1, T)` with `T ≫ G`).

---

## 7. Provenance of the Cardioformer backbone

**Files:** `models/Cardioformer.py`, `models/CardioformerCKD.py`, `README.md`

The backbone is a clean-room re-implementation from arXiv:2505.05538, not the
authors' code, and has never been validated against their weights. That must be
stated in the thesis: published Cardioformer numbers are **not** comparable to
these, the ECG-only baseline is *"our re-implementation of Cardioformer"*, and a
gap against the paper is expected rather than a failed reproduction. Now recorded
in the module docstrings and the README.

---

## Behavioural changes to be aware of

| change | effect |
|---|---|
| default `lradj` `type1` → `cosine` | old runs are **not** bit-reproducible unless you pass `--lradj type1` |
| default `fusion=cross` now attends over patches | pass `--cross_attn_tokens granularity` for the old behaviour |
| `test()` requires a checkpoint | eval-only runs now fail loudly instead of scoring noise |
| `td_auroc` masked by eligibility | this number will **go down** relative to previously logged values; the earlier one was wrong |
| metrics json gains `n_eligible`, `n_total`, `brier_at_horizon`, `checkpoint`, `setting` | — |

## Test status

`tests/test_config_override.py` and `tests/test_metrics.py` were executed and pass.
The torch-dependent suites (`test_model_forward.py`, `test_checkpoint_and_schedule.py`,
`test_dataloader_shapes.py`, `test_egfr.py`) are unmodified in their existing style
and run under the CI workflow, which installs CPU torch — run `pytest tests/ -q`
locally once before relying on them.
