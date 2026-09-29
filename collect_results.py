"""Aggregate ablation results into thesis-ready tables.

Reads ``<results>/abl_<variant>_s<seed>/test_metrics.json`` for every variant
and seed in the plan and writes, next to them:

    ablation_runs.csv       one row per (variant, seed)
    ablation_summary.csv    mean, std over seeds, and paired delta vs reference
    ablation_summary.md     one Markdown table per ablation group
    ablation_summary.tex    the same as LaTeX booktabs tables

The paired delta is computed seed-by-seed (variant_s - reference_s), so seed
noise shared by both runs cancels; its std shows whether a difference is larger
than run-to-run variation.

    python collect_results.py --plan configs/ablations.yaml [-- --results results/ckd_prognosis]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from run_ablations import load_plan, results_dir_for, build_argv

PRETTY = {
    "c_index": "C-index", "td_auroc_365d": "AUROC@1y", "td_auroc_730d": "AUROC@2y",
    "brier_ipcw_730d": "Brier@2y", "brier_ipcw_365d": "Brier@1y", "ece_730d": "ECE@2y",
    "hr_high_vs_low": "HR high/low",
}
LOWER_IS_BETTER = {"brier_ipcw_730d", "brier_ipcw_365d", "ece_730d"}


def _fmt(mean, std, n):
    if mean is None or not np.isfinite(mean):
        return "--"
    return f"{mean:.3f} ± {std:.3f}" if n > 1 and np.isfinite(std) else f"{mean:.3f}"


def _fmt_delta(d, sd, n):
    if d is None or not np.isfinite(d):
        return "--"
    s = f"{d:+.3f}"
    return f"{s} ± {sd:.3f}" if n > 1 and np.isfinite(sd) else s


def collect(plan_path, forwarded=()):
    plan = load_plan(plan_path)
    metrics = plan.get("report_metrics", ["c_index"])
    ref = plan["reference"]
    runs = {}
    rows = []
    res_root = None
    for v in plan["variants"]:
        for s in plan["seeds"]:
            out = results_dir_for(build_argv(plan, v, s, forwarded))
            res_root = res_root or os.path.dirname(out)
            f = os.path.join(out, "test_metrics.json")
            if not os.path.exists(f):
                continue
            m = json.load(open(f))
            runs[(v, s)] = m
            rows.append(dict({"variant": v, "seed": s},
                             **{k: m.get(k, float("nan")) for k in metrics}))

    summary = []
    for v in plan["variants"]:
        seeds_v = [s for s in plan["seeds"] if (v, s) in runs]
        if not seeds_v:
            continue
        r = {"variant": v, "n_seeds": len(seeds_v)}
        for k in metrics:
            vals = np.array([float(runs[(v, s)].get(k, np.nan)) for s in seeds_v])
            vals = vals[np.isfinite(vals)]
            r[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
            r[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
            paired = [float(runs[(v, s)].get(k, np.nan)) - float(runs[(ref, s)].get(k, np.nan))
                      for s in seeds_v if (ref, s) in runs]
            paired = np.array([d for d in paired if np.isfinite(d)])
            r[f"{k}_delta"] = float(paired.mean()) if paired.size else float("nan")
            r[f"{k}_delta_std"] = float(paired.std(ddof=1)) if paired.size > 1 else float("nan")
        summary.append(r)
    return plan, rows, summary, res_root


def write_tables(plan, rows, summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metrics = plan.get("report_metrics", ["c_index"])
    primary = plan.get("primary_metric", metrics[0])
    by_v = {r["variant"]: r for r in summary}

    for name, data in (("ablation_runs.csv", rows), ("ablation_summary.csv", summary)):
        if data:
            with open(os.path.join(out_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)

    md, tex = ["# Ablation study results", "",
               f"Mean ± std over seeds {plan['seeds']}. Δ = paired difference vs "
               f"`{plan['reference']}` on the primary metric ({PRETTY.get(primary, primary)}).", ""], []
    for g, vs in plan["groups"].items():
        head = ["Variant", "n"] + [PRETTY.get(k, k) for k in metrics] + [f"Δ {PRETTY.get(primary, primary)}"]
        md += [f"## {g}", "", "| " + " | ".join(head) + " |",
               "|" + "|".join(["---"] * len(head)) + "|"]
        tex += [f"% ---- {g}", "\\begin{table}[h]\\centering\\small",
                "\\begin{tabular}{l" + "c" * (len(head) - 1) + "}", "\\toprule",
                " & ".join(head).replace("±", "$\\pm$").replace("Δ", "$\\Delta$") + " \\\\", "\\midrule"]
        for v in vs:
            r = by_v.get(v)
            if r is None:
                cells = [v, "0"] + ["--"] * (len(metrics) + 1)
            else:
                n = r["n_seeds"]
                cells = [v, str(n)] + [_fmt(r[f"{k}_mean"], r[f"{k}_std"], n) for k in metrics]
                cells.append("ref" if v == plan["reference"]
                             else _fmt_delta(r[f"{primary}_delta"], r[f"{primary}_delta_std"], n))
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(c.replace("_", "\\_").replace("±", "$\\pm$") for c in cells) + " \\\\")
        md.append("")
        tex += ["\\bottomrule", "\\end{tabular}",
                f"\\caption{{Ablation: {g.replace('_', ' ')}.}}", "\\end{table}", ""]
    open(os.path.join(out_dir, "ablation_summary.md"), "w").write("\n".join(md))
    open(os.path.join(out_dir, "ablation_summary.tex"), "w").write("\n".join(tex))
    return os.path.join(out_dir, "ablation_summary.md")


def main(argv=None):
    p = argparse.ArgumentParser(description="Collect ablation results")
    p.add_argument("--plan", default="configs/ablations.yaml")
    p.add_argument("--out_dir", default=None, help="default: the results root")
    argv = list(sys.argv[1:] if argv is None else argv)
    forwarded = []
    if "--" in argv:
        i = argv.index("--")
        argv, forwarded = argv[:i], argv[i + 1:]
    args, unknown = p.parse_known_args(argv)
    plan, rows, summary, res_root = collect(args.plan, unknown + forwarded)
    if not rows:
        print("[collect] no finished runs found")
        return 1
    path = write_tables(plan, rows, summary, args.out_dir or res_root)
    print(open(path).read())
    print(f"[collect] {len(rows)} runs -> {os.path.dirname(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
