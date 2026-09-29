"""Thesis figures from a finished run / ablation study (matplotlib).

    python scripts/plot_results.py --run results/ckd_prognosis/ckd_2y_survival_multimodal
    python scripts/plot_results.py --ablation results/ckd_prognosis/ablation_summary.csv

Produces, in the run directory (or next to the summary):
    km_by_risk_group.png   observed cumulative CKD incidence per risk group
                           (Kaplan-Meier), with the log-rank p-value and HR
    calibration.png        predicted vs observed risk by decile at the horizon
    risk_distribution.png  predicted absolute risk, events vs non-events
    ablation_<metric>.png  paired delta vs the reference model, per variant
"""

import argparse
import csv
import json
import os

import numpy as np


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_run(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = json.load(open(os.path.join(run_dir, "risk_groups.json")))
    km = _read_csv(os.path.join(run_dir, "km_curves.csv"))
    days = np.array([float(r["day"]) for r in km])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in [k for k in km[0] if k != "day"]:
        n = next((g["n"] for g in groups["groups"] if g["group"] == name), None)
        ax.step(days / 365.25, [float(r[name]) for r in km], where="post",
                label=f"{name} risk (n={n})")
    ax.set_xlabel("Years since index ECG")
    ax.set_ylabel("Cumulative incidence of CKD")
    p = groups.get("logrank_p", float("nan"))
    ax.set_title(f"Log-rank p = {p:.2g};  HR high vs low = {groups['hr_high_vs_low']:.2f} "
                 f"[{groups['hr_lo']:.2f}, {groups['hr_hi']:.2f}]", fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "km_by_risk_group.png"), dpi=200)
    plt.close(fig)

    cal_path = os.path.join(run_dir, "calibration.csv")
    if os.path.exists(cal_path):
        cal = _read_csv(cal_path)
        pr = np.array([float(r["mean_predicted"]) for r in cal])
        ob = np.array([float(r["observed_km"]) for r in cal])
        lo = np.array([float(r["observed_lo"]) for r in cal])
        hi = np.array([float(r["observed_hi"]) for r in cal])
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        lim = max(pr.max(), hi.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect calibration")
        ax.errorbar(pr, ob, yerr=[ob - lo, hi - ob], fmt="o", capsize=3, label="deciles")
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed risk (Kaplan-Meier)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "calibration.png"), dpi=200)
        plt.close(fig)

    preds = _read_csv(os.path.join(run_dir, "test_predictions.csv"))
    col = [c for c in preds[0] if c.startswith("abs_risk_")][-1]
    r = np.array([float(x[col]) for x in preds])
    ev = np.array([int(x["event_by_horizon"]) for x in preds]).astype(bool)
    if np.isfinite(r).all():
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bins = np.linspace(0, max(r.max(), 1e-3), 30)
        ax.hist(r[~ev], bins=bins, alpha=0.6, density=True, label="no CKD by horizon")
        ax.hist(r[ev], bins=bins, alpha=0.6, density=True, label="incident CKD by horizon")
        for t in groups["thresholds"]["thresholds"] if groups["thresholds"]["on"] == "abs_risk" else []:
            ax.axvline(t, color="k", ls=":")
        ax.set_xlabel(f"Predicted {col.replace('abs_risk_', '')} risk")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "risk_distribution.png"), dpi=200)
        plt.close(fig)
    print(f"[plot] figures written to {run_dir}")


def plot_ablation(summary_csv, metric="c_index"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(summary_csv)
    rows = [r for r in rows if np.isfinite(float(r[f"{metric}_delta"]))]
    names = [r["variant"] for r in rows]
    d = np.array([float(r[f"{metric}_delta"]) for r in rows])
    sd = np.array([float(r[f"{metric}_delta_std"]) if r[f"{metric}_delta_std"] not in ("", "nan")
                   else 0.0 for r in rows])
    sd = np.nan_to_num(sd)
    order = np.argsort(d)
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(rows) + 1))
    ax.barh(np.array(names)[order], d[order], xerr=sd[order], capsize=2,
            color=["C3" if x < 0 else "C2" for x in d[order]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"Δ {metric} vs reference (paired over seeds)")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(summary_csv), f"ablation_{metric}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="results/<...>/<setting> directory")
    p.add_argument("--ablation", default=None, help="ablation_summary.csv")
    p.add_argument("--metric", default="c_index")
    a = p.parse_args()
    if a.run:
        plot_run(a.run)
    if a.ablation:
        plot_ablation(a.ablation, a.metric)
