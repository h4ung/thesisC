"""Ablation plumbing + an end-to-end train/test run on synthetic data (CPU, ~10 s)."""

import json
import os

import pytest
import torch

from run import resolve_args
from run_ablations import build_argv, load_plan, overrides_to_argv, selected_variants

PLAN = os.path.join(os.path.dirname(__file__), "..", "configs", "ablations.yaml")
ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_plan_is_valid_and_every_group_contains_the_reference():
    plan = load_plan(PLAN)
    for g, vs in plan["groups"].items():
        assert plan["reference"] in vs, f"group {g} has no reference model to compare to"


def test_variant_overrides_become_real_cli_flags():
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        plan = load_plan(PLAN)
        a = resolve_args(build_argv(plan, "ecg_only", 42, ["--use_ehr", "1"]))
        assert a.use_ehr is False                  # variant beats forwarded flag
        assert a.setting == "abl_ecg_only_s42" and a.seed == 42
        b = resolve_args(build_argv(plan, "linear_cox_ehr", 41, []))
        assert (b.head, b.head_hidden, b.ehr_mode, b.use_ecg) == ("cox", 0, "identity", False)
        c = resolve_args(build_argv(plan, "no_inter_granularity", 41, []))
        assert c.inter_granularity is False
    finally:
        os.chdir(cwd)


def test_overrides_to_argv_bools_and_lists():
    assert overrides_to_argv({"augment": False, "x": [1, 2]}) == ["--augment", "0", "--x", "1,2"]


def test_group_selection_deduplicates_reference():
    plan = load_plan(PLAN)
    names = selected_variants(plan, groups="modality,fusion")
    assert names.count("full") == 1


def test_backbone_ablation_switches_forward():
    from models.CardioformerCKD import CardioformerCKD
    from tests.test_model_forward import Cfg
    for kw in ({"patch_encoder": "linear"}, {"inter_granularity": False},
               {"cross_channel": False, "patch_encoder": "linear"}):
        cfg = Cfg()
        for k, v in kw.items():
            setattr(cfg, k, v)
        m = CardioformerCKD(cfg)
        out = m(x_ecg=torch.randn(2, cfg.seq_len, cfg.enc_in), ehr=torch.randn(2, cfg.ehr_in_dim))
        assert out.shape == (2, cfg.n_intervals)


@pytest.mark.parametrize("head", ["survival", "cox"])
def test_end_to_end_train_test_writes_risk_outputs(tmp_path, head):
    from run import main
    from scripts.make_synthetic_dataset import make
    data = tmp_path / "syn"
    make(str(data), n_subjects=80, seed=2)
    argv = ["--root_path", str(data), "--setting", f"e2e_{head}", "--head", head,
            "--checkpoints", str(tmp_path / "ck"), "--results", str(tmp_path / "res"),
            "--use_gpu", "0", "--num_workers", "0", "--d_model", "16", "--n_heads", "2",
            "--e_layers", "1", "--d_ff", "32", "--patch_len_list", "32",
            "--resnet_hidden", "8", "--resnet_blocks", "1", "--d_fuse", "32",
            "--train_epochs", "1", "--batch_size", "32", "--n_bootstrap", "5",
            "--seq_len", "250"]
    main(argv)
    out = tmp_path / "res" / f"e2e_{head}"
    for f in ("test_metrics.json", "test_predictions.csv", "val_predictions.csv",
              "risk_groups.json", "km_curves.csv", "calibration.csv"):
        assert (out / f).exists(), f
    m = json.load(open(out / "test_metrics.json"))
    for k in ("c_index", "td_auroc_365d", "td_auroc_730d", "brier_ipcw_730d",
              "hr_high_vs_low", "observed_risk_low", "observed_risk_high", "c_index_ci95"):
        assert k in m
    g = json.load(open(out / "risk_groups.json"))
    assert [x["group"] for x in g["groups"]] == ["low", "high"]
    # eval-only reload reproduces the same numbers (checkpoint + scaler + baseline)
    main(argv + ["--is_training", "0"])
    m2 = json.load(open(out / "test_metrics.json"))
    assert m2["c_index"] == m["c_index"] and m2["td_auroc_730d"] == m["td_auroc_730d"]


def test_make_splits_is_subject_independent_under_pandas3(tmp_path):
    """pandas 3 copy-on-write makes .to_numpy() read-only; shuffling it in place
    used to raise 'array is read-only' (CI failure on the 3.11 job)."""
    import argparse

    import pandas as pd

    from data_preprocessing.make_splits import run
    df = pd.DataFrame({"subject_id": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10],
                       "study_id": range(12),
                       "label_binary": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1]})
    src = tmp_path / "ds.parquet"
    df.to_parquet(src)
    run(argparse.Namespace(dataset=str(src), out_dir=str(tmp_path), train_frac=0.6,
                           val_frac=0.2, seed=41))
    out = pd.read_parquet(tmp_path / "ckd_dataset_split.parquet")
    assert out["split"].notna().all()
    assert (out.groupby("subject_id")["split"].nunique() == 1).all()   # no subject leaks
