"""Entry point for CKD prognosis experiments.

Example
-------
python run.py \
    --task_name ckd_prognosis --is_training 1 \
    --root_path dataset/ckd --data MIMICIV_CKD \
    --model CardioformerCKD \
    --enc_in 12 --seq_len 250 \
    --d_model 128 --n_heads 8 --e_layers 6 --d_ff 256 \
    --patch_len_list 2,4,8,8,16,16,16,16,32 \
    --head survival --n_intervals 8 --horizon_days 730 --fusion gated \
    --use_ecg 1 --use_ehr 1 \
    --batch_size 16 --learning_rate 1e-4 --train_epochs 20 --patience 5

Configuration precedence
------------------------
    explicit CLI flag  >  --config YAML  >  argparse default

Implemented in ``resolve_args``: a second, defaults-suppressed parser discovers
exactly which flags the user typed, and YAML values are only written into keys
the user did *not* pass. The previous implementation applied the YAML *after*
parsing, so YAML silently overrode the CLI -- which meant that e.g.
``--config configs/ckd_prognosis.yaml --use_ehr 0`` did not actually ablate the
EHR branch.

``resolve_args`` imports nothing heavy, so it is unit-testable without torch
(see tests/test_config_override.py).
"""

import argparse
import os
import sys

import yaml


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y")


def build_parser():
    p = argparse.ArgumentParser(description="CardioformerCKD - prognostic CKD prediction")

    # basic
    p.add_argument("--task_name", default="ckd_prognosis")
    p.add_argument("--is_training", type=int, default=1)
    p.add_argument("--model", default="CardioformerCKD")
    p.add_argument("--setting", default="ckd_run")
    p.add_argument("--config", default=None, help="optional YAML config path")

    # data
    p.add_argument("--root_path", default="dataset/ckd")
    p.add_argument("--data", default="MIMICIV_CKD")
    p.add_argument("--seq_len", type=int, default=250, help="ECG window length (timestamps)")
    p.add_argument("--enc_in", type=int, default=12, help="number of ECG leads")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--augment", type=str2bool, default=True)

    # Cardioformer backbone
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch_len_list", default="2,4,8,8,16,16,16,16,32")
    p.add_argument("--resnet_hidden", type=int, default=64)
    p.add_argument("--resnet_blocks", type=int, default=2)
    p.add_argument("--cross_channel", type=str2bool, default=True)
    p.add_argument("--patch_encoder", choices=["resnet", "linear"], default="resnet",
                   help="patch -> token encoder: Cardioformer's 1-D ResNet (default) or a "
                        "flatten+linear projection (ablation)")
    p.add_argument("--inter_granularity", type=str2bool, default=True,
                   help="stage-2 inter-granularity attention in every encoder layer "
                        "(set 0 for the ablation)")

    # multimodal / prognostic
    p.add_argument("--use_ecg", type=str2bool, default=True)
    p.add_argument("--use_ehr", type=str2bool, default=True)
    p.add_argument("--ehr_mode", choices=["mlp", "seq", "identity"], default="mlp",
                   help="'identity' feeds standardised features straight to the head "
                        "(with --head cox --head_hidden 0 --use_ecg 0: linear Cox PH baseline)")
    p.add_argument("--ehr_dropout", type=float, default=0.2)
    p.add_argument("--ehr_drop_groups", default="",
                   help="comma-separated EHR feature groups to remove (ablation): "
                        "demographics, renal, labs_other, comorbidities, or raw column prefixes")
    p.add_argument("--fusion", choices=["concat", "gated", "cross"], default="gated")
    p.add_argument("--cross_attn_tokens", choices=["patches", "granularity"], default="patches",
                   help="what cross-attention fusion attends over: the full multi-granularity patch "
                        "sequence (default) or only the G pooled granularity summaries (legacy)")
    p.add_argument("--d_fuse", type=int, default=256)
    # prognostic RISK layer (Thesis C)
    p.add_argument("--head", choices=["survival", "cox", "binary"], default="survival",
                   help="risk-prediction head: 'survival' discrete-time hazards (default), "
                        "'cox' proportional-hazards log-risk + Breslow baseline, or "
                        "'binary' fixed-horizon classifier (ablation baseline only)")
    p.add_argument("--head_hidden", type=int, default=128, help="head MLP width; 0 = linear head")
    p.add_argument("--head_dropout", type=float, default=0.2)
    p.add_argument("--n_intervals", type=int, default=8, help="survival time bins over horizon")
    p.add_argument("--horizon_days", type=float, default=730,
                   help="prediction horizon; MUST match build_cohort --horizon_days. Events "
                        "after it are censored at it for training and evaluation")
    p.add_argument("--eval_horizons_days", default="365,730",
                   help="horizons (days, <= horizon_days) at which absolute risk, td-AUROC "
                        "and Brier are reported")

    # risk stratification (high / low risk groups)
    p.add_argument("--risk_strat_method", choices=["quantile", "absolute", "youden", "sensitivity"],
                   default="quantile",
                   help="how validation-set cut-points are chosen (see utils/risk_stratification.py)")
    p.add_argument("--risk_strat_quantiles", default="0.5",
                   help="quantile method: '0.5' = low/high median split, '0.33,0.67' = tertiles")
    p.add_argument("--risk_strat_cutoffs", default="0.05,0.15",
                   help="absolute method: predicted-risk cut-points at the strat horizon")
    p.add_argument("--risk_strat_target_sens", type=float, default=0.8,
                   help="sensitivity method: minimum validation sensitivity for 'high risk'")
    p.add_argument("--risk_strat_horizon_days", type=float, default=None,
                   help="horizon used for risk groups (default: --horizon_days)")

    # evaluation
    p.add_argument("--n_bootstrap", type=int, default=200,
                   help="subject-level bootstrap replicates for test CIs (0 = off)")
    p.add_argument("--calibration_bins", type=int, default=10)
    p.add_argument("--early_stop_metric", choices=["c_index", "td_auroc"], default="c_index")

    # optimisation
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lradj", choices=["cosine", "step", "type1", "none"], default="cosine",
                   help="LR schedule. 'cosine' (default) anneals over train_epochs; 'step' decays "
                        "by --lr_decay_rate every --lr_decay_every epochs; 'type1' is the legacy "
                        "halve-every-epoch schedule (kept for reproducibility; not recommended)")
    p.add_argument("--lr_decay_every", type=int, default=5, help="epochs per decay step (lradj=step)")
    p.add_argument("--lr_decay_rate", type=float, default=0.5, help="decay factor (lradj=step)")
    p.add_argument("--min_lr", type=float, default=1e-6, help="floor for the LR schedule")
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=41)

    # system
    p.add_argument("--use_gpu", type=str2bool, default=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--devices", default="0")
    p.add_argument("--checkpoints", default="checkpoints/ckd_prognosis")
    p.add_argument("--checkpoint_path", default=None,
                   help="explicit checkpoint .pth to load "
                        "(default: <checkpoints>/<setting>/checkpoint.pth)")
    p.add_argument("--results", default="results/ckd_prognosis")
    return p


def _explicit_cli_keys(argv):
    """Return the set of dests the user actually passed on the command line.

    Re-build the same parser with every default suppressed, so the resulting
    namespace contains *only* the flags that appeared in ``argv``.
    """
    p = build_parser()
    for action in p._actions:
        if action.dest != "help":
            action.default = argparse.SUPPRESS
    ns, _ = p.parse_known_args(argv)
    return set(vars(ns).keys())


def _coerce(parser, key, value):
    """Cast a YAML value with the same ``type=`` callable argparse would use."""
    action = next((a for a in parser._actions if a.dest == key), None)
    if action is None or action.type is None or isinstance(value, (list, dict)):
        return value
    try:
        return action.type(value)
    except (TypeError, ValueError):
        return value


def resolve_args(argv=None):
    """Parse CLI + YAML with precedence: explicit CLI > YAML > argparse default."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    passed = _explicit_cli_keys(argv)

    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"--config not found: {args.config}")
        cfg = yaml.safe_load(open(args.config)) or {}
        known = {a.dest for a in parser._actions}
        unknown = sorted(set(cfg) - known)
        if unknown:
            print(f"[config] warning: ignoring unknown key(s) in {args.config}: {', '.join(unknown)}")
        overridden = []
        for k, v in cfg.items():
            if k not in known or k == "config":
                continue
            if k in passed:                      # explicit CLI flag wins
                overridden.append(k)
                continue
            setattr(args, k, _coerce(parser, k, v))
        if overridden:
            print(f"[config] CLI overrides YAML for: {', '.join(sorted(overridden))}")

    args.cli_overrides = sorted(passed - {"config"})
    return args


def main(argv=None):
    args = resolve_args(argv)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.devices)
    print("Args:", vars(args))

    # imported late so resolve_args() stays importable without torch installed
    from exp.exp_ckd_prognosis import Exp_CKD_Prognosis

    exp = Exp_CKD_Prognosis(args)
    if args.is_training:
        exp.train()
    exp.test()


if __name__ == "__main__":
    main()
