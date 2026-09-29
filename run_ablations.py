"""Run the ablation study defined in an ablation plan YAML (configs/ablations.yaml).

Each (variant, seed) is a separate ``run.py`` process -- fresh memory, fresh RNG
-- with setting name ``abl_<variant>_s<seed>``. Any extra arguments after the
known ones are forwarded to every run (e.g. data location, device, a shorter
epoch budget for a smoke test):

    python run_ablations.py --plan configs/ablations.yaml --groups modality \
        -- --root_path dataset/ckd --use_gpu 1

Variant overrides are applied *after* forwarded arguments, so a forwarded flag
can never silently undo the component being ablated.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_plan(path):
    plan = yaml.safe_load(open(path))
    for k in ("base_config", "variants", "groups"):
        if k not in plan:
            raise ValueError(f"ablation plan {path} is missing '{k}'")
    plan.setdefault("reference", "full")
    plan.setdefault("seeds", [41])
    for g, vs in plan["groups"].items():
        for v in vs:
            if v not in plan["variants"]:
                raise ValueError(f"group '{g}' lists unknown variant '{v}'")
    return plan


def selected_variants(plan, groups=None, variants=None):
    if variants:
        names = [v.strip() for v in variants.split(",") if v.strip()]
    else:
        gs = [g.strip() for g in groups.split(",")] if groups else list(plan["groups"])
        names = []
        for g in gs:
            if g not in plan["groups"]:
                raise ValueError(f"unknown group '{g}'; plan has {list(plan['groups'])}")
            names += [v for v in plan["groups"][g] if v not in names]
    for v in names:
        if v not in plan["variants"]:
            raise ValueError(f"unknown variant '{v}'")
    return names


def setting_name(variant, seed):
    return f"abl_{variant}_s{seed}"


def overrides_to_argv(overrides):
    argv = []
    for k, v in (overrides or {}).items():
        if isinstance(v, bool):
            v = "1" if v else "0"
        elif isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v)
        argv += [f"--{k}", str(v)]
    return argv


def build_argv(plan, variant, seed, forwarded):
    return (["--config", plan["base_config"]] + list(forwarded)
            + overrides_to_argv(plan["variants"][variant])
            + ["--setting", setting_name(variant, seed), "--seed", str(seed),
               "--is_training", "1"])


def results_dir_for(argv):
    import contextlib
    import io
    sys.path.insert(0, HERE)
    from run import resolve_args
    with contextlib.redirect_stdout(io.StringIO()):   # silence the [config] notices
        a = resolve_args(argv)
    return os.path.join(a.results, a.setting)


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the CKD risk-model ablation study")
    p.add_argument("--plan", default="configs/ablations.yaml")
    p.add_argument("--groups", default=None, help="comma-separated groups (default: all)")
    p.add_argument("--variants", default=None, help="comma-separated variants (overrides --groups)")
    p.add_argument("--seeds", default=None, help="comma-separated seeds (default: from plan)")
    p.add_argument("--skip_existing", type=int, default=1,
                   help="skip runs whose test_metrics.json already exists")
    p.add_argument("--dry_run", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        i = argv.index("--")
        argv, forwarded = argv[:i], argv[i + 1:]
    else:
        forwarded = []
    args, unknown = p.parse_known_args(argv)
    forwarded = unknown + forwarded

    plan = load_plan(args.plan)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else plan["seeds"])
    names = selected_variants(plan, args.groups, args.variants)
    print(f"[ablation] {len(names)} variants x {len(seeds)} seeds = {len(names) * len(seeds)} runs")

    manifest = []
    for v in names:
        for s in seeds:
            run_argv = build_argv(plan, v, s, forwarded)
            out = results_dir_for(run_argv)
            done = os.path.exists(os.path.join(out, "test_metrics.json"))
            cmd = [sys.executable, os.path.join(HERE, "run.py")] + run_argv
            entry = {"variant": v, "seed": s, "setting": setting_name(v, s),
                     "results_dir": out, "cmd": " ".join(cmd)}
            if args.dry_run:
                print(entry["cmd"])
            elif done and args.skip_existing:
                print(f"[ablation] skip {entry['setting']} (results exist)")
                entry["status"] = "skipped"
            else:
                print(f"[ablation] >>> {entry['setting']}")
                t0 = time.time()
                rc = subprocess.call(cmd, cwd=HERE)
                entry["status"] = "ok" if rc == 0 else f"failed({rc})"
                entry["seconds"] = round(time.time() - t0, 1)
                if rc != 0:
                    print(f"[ablation] !!! {entry['setting']} failed with exit code {rc}")
            manifest.append(entry)

    if not args.dry_run and manifest:
        res_root = os.path.dirname(manifest[0]["results_dir"])
        os.makedirs(res_root, exist_ok=True)
        path = os.path.join(res_root, "ablation_manifest.json")
        json.dump(manifest, open(path, "w"), indent=2)
        n_fail = sum(1 for m in manifest if str(m.get("status", "")).startswith("failed"))
        print(f"[ablation] manifest -> {path} ({n_fail} failed)")
        return 1 if n_fail else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
