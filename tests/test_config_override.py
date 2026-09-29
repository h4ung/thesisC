"""Precedence: explicit CLI flag > --config YAML > argparse default.

These guard the bug where apply_yaml() overwrote the parsed namespace, so
`--config ... --use_ehr 0` silently kept use_ehr=True and the "ablations" did
not ablate anything.

run.py imports torch only inside main(), so these run on a bare CI box.
"""

import textwrap

import pytest

from run import resolve_args

YAML = textwrap.dedent("""
    setting: from_yaml
    use_ecg: true
    use_ehr: true
    fusion: gated
    head: survival
    d_model: 128
    learning_rate: 0.0001
    train_epochs: 30
    lradj: cosine
    made_up_key: 123
""")


@pytest.fixture()
def cfg(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(YAML)
    return str(p)


def test_yaml_fills_unset_keys(cfg):
    a = resolve_args(["--config", cfg])
    assert a.setting == "from_yaml"
    assert a.train_epochs == 30
    assert a.use_ehr is True


def test_cli_beats_yaml_for_ablation_flags(cfg):
    a = resolve_args(["--config", cfg, "--use_ehr", "0"])
    assert a.use_ehr is False          # the ablation actually ablates
    assert a.use_ecg is True           # untouched key still comes from YAML


def test_cli_beats_yaml_for_setting_and_scalars(cfg):
    a = resolve_args(["--config", cfg, "--setting", "ckd_2y_ecg_only",
                      "--learning_rate", "1e-3", "--d_model", "64"])
    assert a.setting == "ckd_2y_ecg_only"   # ablations keep separate checkpoint dirs
    assert a.learning_rate == 1e-3
    assert a.d_model == 64


def test_cli_value_equal_to_default_is_still_respected(cfg):
    # --head survival is also the parser default; passing it explicitly must not
    # be mistaken for "unset" and then re-read from YAML.
    a = resolve_args(["--config", cfg, "--head", "binary"])
    assert a.head == "binary"
    assert "head" in a.cli_overrides


def test_yaml_values_are_type_coerced(cfg):
    a = resolve_args(["--config", cfg])
    assert isinstance(a.d_model, int)
    assert isinstance(a.learning_rate, float)
    assert isinstance(a.use_ecg, bool)


def test_defaults_used_when_no_yaml():
    a = resolve_args([])
    assert a.setting == "ckd_run"
    assert a.lradj == "cosine"          # not the legacy halve-every-epoch schedule


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        resolve_args(["--config", "does/not/exist.yaml"])
