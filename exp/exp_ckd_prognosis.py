"""Experiment for prognostic CKD **risk prediction** with CardioformerCKD.

Thesis C: the prognostic layer predicts *risk*, not a binary class.

For every patient the experiment produces
    risk_score        continuous ranking score (higher = CKD expected sooner)
    abs_risk_<h>d     predicted probability of incident CKD by each evaluation
                      horizon h (``--eval_horizons_days``, e.g. 1 y and 2 y)
    risk_group        low / high (or low / intermediate / high) risk, using
                      cut-points fitted on the VALIDATION split only

and evaluates
    discrimination    Harrell's C-index (truncated at the horizon), time-dependent
                      AUROC at each horizon (horizon-eligible patients only)
    calibration       IPCW Brier score at each horizon, calibration-in-deciles
                      table and expected calibration error (ECE)
    stratification    observed (Kaplan-Meier) CKD incidence per risk group,
                      log-rank test, hazard ratio high vs low
    uncertainty       subject-level bootstrap 95% CIs on the headline metrics

Outputs (``<results>/<setting>/``)
    test_metrics.json, test_predictions.csv, val_predictions.csv,
    risk_groups.json, km_curves.csv, calibration.csv
"""

import csv
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim

from data_provider.data_factory import data_provider
from data_provider.data_loader import EHRStandardizer
from exp.exp_basic import Exp_Basic
from models.CardioformerCKD import CardioformerCKD
from models.heads import BreslowBaseline, DiscreteTimeSurvivalHead, horizon_bin_index
from utils.losses import cox_ph_loss, discrete_time_nll, focal_bce
from utils.metrics import (
    bootstrap_ci, concordance_index, horizon_eligibility, time_dependent_auroc,
)
from utils.risk_stratification import (
    assign_groups, fit_thresholds, group_names, summarise_groups,
)
from utils.survival import calibration_table, ipcw_brier_score
from utils.tools import EarlyStopping, adjust_learning_rate, set_seed


def _parse_horizons(s, default):
    if s is None or s == "":
        return [float(default)]
    if isinstance(s, (int, float)):
        return [float(s)]
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def _hkey(h):
    return f"{int(round(h))}d"


class Exp_CKD_Prognosis(Exp_Basic):
    def __init__(self, args):
        set_seed(args.seed)
        self.horizon = float(getattr(args, "horizon_days", 730))
        self.eval_horizons = _parse_horizons(getattr(args, "eval_horizons_days", None),
                                             self.horizon)
        if self.horizon not in self.eval_horizons:
            self.eval_horizons.append(self.horizon)
        self.eval_horizons = sorted(self.eval_horizons)
        self.strat_horizon = float(getattr(args, "risk_strat_horizon_days", None) or self.horizon)
        self.baseline = None  # Breslow baseline hazard (cox head only)

        # discover ehr_in_dim from the train dataset before building the model
        self.train_data, self.train_loader = data_provider(args, "train")
        self._scaler = self.train_data.scaler
        self._ehr_columns = self.train_data.ehr_columns
        self._check_bins(args)

        # For eval-only runs, prefer the scaler persisted alongside the checkpoint
        # so preprocessing at test time is byte-identical to training.
        if not args.is_training:
            self._load_saved_scaler(os.path.join(args.checkpoints, args.setting,
                                                 "ehr_scaler.json"))

        args.ehr_in_dim = len(self._ehr_columns)
        super().__init__(args)

    # ---- setup helpers ---------------------------------------------------
    def _check_bins(self, args):
        if args.head != "survival":
            return
        max_bin = int(self.train_data.df["event_bin"].max()) if len(self.train_data) else 0
        if max_bin >= args.n_intervals:
            raise ValueError(
                f"cohort was built with more survival bins (max event_bin={max_bin}) than "
                f"--n_intervals={args.n_intervals}; rebuild the cohort or match n_intervals")
        for h in self.eval_horizons:
            horizon_bin_index(h, self.horizon, args.n_intervals)   # raises if h > horizon

    def _load_saved_scaler(self, path):
        if not os.path.exists(path):
            return False
        state = json.load(open(path))
        self._scaler = EHRStandardizer().load_state_dict(state)
        self._ehr_columns = list(state["columns_"])
        print(f"[scaler] loaded fitted EHR scaler from {path}")
        return True

    def _build_model(self):
        return CardioformerCKD(self.args)

    def _get_data(self, flag, deterministic=False):
        return data_provider(self.args, flag, scaler=self._scaler,
                             ehr_columns=self._ehr_columns, deterministic=deterministic)

    def _baseline_file(self):
        return os.path.join(os.path.dirname(self.checkpoint_file()), "baseline_hazard.json")

    # ---- batching helpers ----------------------------------------------
    def _to_device(self, batch):
        x_ecg = batch.get("x_ecg")
        ehr = batch.get("ehr")
        if x_ecg is not None:
            x_ecg = x_ecg.float().to(self.device)
        if ehr is not None:
            ehr = ehr.float().to(self.device)
        return x_ecg, ehr

    def _compute_loss(self, out, batch):
        head = self.args.head
        if head == "binary":
            y = batch["label_binary"].to(self.device)
            elig = batch["eligible_binary"].to(self.device).bool()
            if elig.sum() == 0:
                return out.sum() * 0.0
            return focal_bce(out[elig], y[elig],
                             alpha=self.args.focal_alpha, gamma=self.args.focal_gamma)
        if head == "cox":
            return cox_ph_loss(out, batch["event_time_h"].to(self.device),
                               batch["event_indicator_h"].to(self.device))
        # survival: horizon-censored indicator (events after the horizon are censored)
        return discrete_time_nll(out, batch["event_bin"].to(self.device),
                                 batch["event_indicator_h"].to(self.device))

    # ---- main loops -----------------------------------------------------
    def train(self):
        train_loader = self.train_loader
        _, vali_loader = self._get_data("val")

        path = self.checkpoint_dir()
        os.makedirs(path, exist_ok=True)
        if self._scaler is not None:
            json.dump(self._scaler.state_dict(),
                      open(os.path.join(path, "ehr_scaler.json"), "w"))
        json.dump({k: v for k, v in vars(self.args).items()
                   if isinstance(v, (int, float, str, bool, list, type(None)))},
                  open(os.path.join(path, "args.json"), "w"), indent=2)

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                               weight_decay=self.args.weight_decay)
        early = EarlyStopping(patience=self.args.patience, mode="max")

        for epoch in range(self.args.train_epochs):
            self.model.train()
            t0 = time.time()
            losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                x_ecg, ehr = self._to_device(batch)
                out = self.model(x_ecg=x_ecg, ehr=ehr)
                loss = self._compute_loss(out, batch)
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
                    optimizer.step()
                losses.append(loss.item())

            val_score, val_metrics = self.validate(vali_loader)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1} | lr {lr_now:.2e} | train_loss {np.mean(losses):.4f} | "
                  f"val_{self.args.early_stop_metric} {val_score:.4f} | {val_metrics} | "
                  f"{time.time()-t0:.1f}s")

            early(val_score, self.model, path)
            if early.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 2, self.args)  # LR for the *next* epoch

        # restore the best weights before returning / testing
        self.load_checkpoint(os.path.join(path, "checkpoint.pth"))
        if self.args.head == "cox":
            self.fit_baseline()
        return self.model

    # ---- inference -------------------------------------------------------
    @torch.no_grad()
    def predict(self, loader):
        """Run the model over ``loader`` and return per-patient arrays."""
        self.model.eval()
        head = self.args.head
        raw, cols = [], {k: [] for k in (
            "subject_id", "study_id", "event_time_days", "event_indicator",
            "event_time_h", "event_indicator_h", "label_binary", "eligible_binary")}
        for batch in loader:
            x_ecg, ehr = self._to_device(batch)
            out = self.model(x_ecg=x_ecg, ehr=ehr)
            if head == "survival":
                _, cif = DiscreteTimeSurvivalHead.survival_from_hazards(out)
                raw.append(cif.cpu().numpy())
            elif head == "cox":
                raw.append(out.cpu().numpy())
            else:
                raw.append(torch.sigmoid(out).cpu().numpy())
            for k in cols:
                cols[k].append(batch[k].numpy())
        pred = {k: np.concatenate(v) if v else np.array([]) for k, v in cols.items()}
        raw = np.concatenate(raw) if raw else np.zeros((0,))
        pred["raw"] = raw
        pred.update(self._risk_from_raw(raw))
        return pred

    def _risk_from_raw(self, raw):
        """Map head output to (risk_score, abs_risk matrix over eval horizons)."""
        head, K = self.args.head, getattr(self.args, "n_intervals", 8)
        n = raw.shape[0]
        abs_risk = np.full((n, len(self.eval_horizons)), np.nan)
        if head == "survival":
            for j, h in enumerate(self.eval_horizons):
                abs_risk[:, j] = raw[:, horizon_bin_index(h, self.horizon, K)]
            score = raw[:, horizon_bin_index(self.strat_horizon, self.horizon, K)]
        elif head == "cox":
            score = raw.reshape(-1)
            if self.baseline is not None and n:
                abs_risk = self.baseline.abs_risk(score, self.eval_horizons)
        else:  # binary: a probability at the horizon only
            score = raw.reshape(-1)
            abs_risk[:, self.eval_horizons.index(self.horizon)] = score
        return {"risk_score": score, "abs_risk": abs_risk}

    def fit_baseline(self):
        """Fit (and persist) the Breslow baseline hazard on the training split."""
        _, loader = self._get_data("train", deterministic=True)
        self.baseline = None
        p = self.predict(loader)
        self.baseline = BreslowBaseline().fit(p["risk_score"], p["event_time_h"],
                                              p["event_indicator_h"])
        json.dump(self.baseline.state_dict(), open(self._baseline_file(), "w"))
        print(f"[cox] Breslow baseline hazard fitted on {len(p['risk_score'])} training rows")

    def _load_or_fit_baseline(self):
        path = self._baseline_file()
        if os.path.exists(path):
            self.baseline = BreslowBaseline().load_state_dict(json.load(open(path)))
            print(f"[cox] loaded baseline hazard from {path}")
        else:
            self.fit_baseline()

    # ---- evaluation -------------------------------------------------------
    def evaluate(self, pred, train_pred=None):
        """Discrimination + calibration metrics from a ``predict`` dict."""
        t_raw, e_raw = pred["event_time_days"], pred["event_indicator"]
        t_h, e_h = pred["event_time_h"], pred["event_indicator_h"]
        score = pred["risk_score"]
        m = {
            # C-index truncated at the horizon (horizon-censored follow-up): the
            # model predicts risk *within* the horizon, so ranking beyond it is
            # not what it was trained to do.
            "c_index": round(float(concordance_index(t_h, score, e_h)), 4),
            # untruncated, for comparability with Thesis B numbers
            "c_index_untruncated": round(float(concordance_index(t_raw, score, e_raw)), 4),
            "n_total": int(len(score)),
            "n_events_by_horizon": int(e_h.sum()),
        }
        # IPCW Brier uses the *raw* follow-up: a patient event-free past h must
        # have T > h to count as a control, which horizon-truncated times hide.
        ct = train_pred["event_time_days"] if train_pred is not None else None
        ce = train_pred["event_indicator"] if train_pred is not None else None
        for j, h in enumerate(self.eval_horizons):
            elig = horizon_eligibility(t_raw, e_raw, h)
            lab = ((e_raw == 1) & (t_raw <= h)).astype(int)
            r = pred["abs_risk"][:, j]
            r_rank = r if np.isfinite(r).all() else score
            m[f"td_auroc_{_hkey(h)}"] = round(float(time_dependent_auroc(r_rank, lab, elig)), 4)
            m[f"n_eligible_{_hkey(h)}"] = int(elig.sum())
            m[f"brier_ipcw_{_hkey(h)}"] = (
                round(ipcw_brier_score(t_raw, e_raw, r, h, ct, ce), 4)
                if np.isfinite(r).all() and len(r) else float("nan"))
        # backward-compatible keys at the primary horizon
        m["td_auroc"] = m[f"td_auroc_{_hkey(self.horizon)}"]
        m["brier_at_horizon"] = m[f"brier_ipcw_{_hkey(self.horizon)}"]
        m["n_eligible"] = m[f"n_eligible_{_hkey(self.horizon)}"]
        return m

    @torch.no_grad()
    def validate(self, loader):
        pred = self.predict(loader)
        m = self.evaluate(pred)
        key = getattr(self.args, "early_stop_metric", "c_index")
        key = "td_auroc" if key == "td_auroc" else "c_index"
        score = m[key]
        return (score if np.isfinite(score) else 0.0), m

    def _bootstrap(self, pred):
        n_boot = int(getattr(self.args, "n_bootstrap", 0) or 0)
        if n_boot <= 0 or len(pred["risk_score"]) == 0:
            return {}
        t_h, e_h, s = pred["event_time_h"], pred["event_indicator_h"], pred["risk_score"]
        t_raw, e_raw = pred["event_time_days"], pred["event_indicator"]
        j = self.eval_horizons.index(self.horizon)
        r = pred["abs_risk"][:, j]
        r = r if np.isfinite(r).all() else s
        H = self.horizon

        def cidx(ix):
            return concordance_index(t_h[ix], s[ix], e_h[ix])

        def auroc(ix):
            el = horizon_eligibility(t_raw[ix], e_raw[ix], H)
            lab = ((e_raw[ix] == 1) & (t_raw[ix] <= H)).astype(int)
            return time_dependent_auroc(r[ix], lab, el)

        out = {}
        for name, fn in (("c_index", cidx), (f"td_auroc_{_hkey(H)}", auroc)):
            lo, hi = bootstrap_ci(fn, pred["subject_id"], n_boot=n_boot,
                                  seed=int(self.args.seed))
            out[f"{name}_ci95"] = [round(lo, 4), round(hi, 4)]
        return out

    def stratify(self, val_pred, test_pred):
        """Fit risk cut-points on validation, apply to test, summarise groups."""
        a = self.args
        j = self.eval_horizons.index(self.strat_horizon) if self.strat_horizon in self.eval_horizons else -1
        val_abs = val_pred["abs_risk"][:, j] if j >= 0 else None
        th = fit_thresholds(
            val_pred["risk_score"], method=a.risk_strat_method,
            quantiles=a.risk_strat_quantiles, cutoffs=a.risk_strat_cutoffs,
            val_label=((val_pred["event_indicator"] == 1) &
                       (val_pred["event_time_days"] <= self.strat_horizon)).astype(int),
            val_eligible=horizon_eligibility(val_pred["event_time_days"],
                                             val_pred["event_indicator"], self.strat_horizon),
            target_sensitivity=a.risk_strat_target_sens,
            val_abs_risk=val_abs,
        )
        if th["on"] == "abs_risk":
            vals_test = test_pred["abs_risk"][:, j]
            if not np.isfinite(vals_test).all():
                raise ValueError("absolute-risk stratification needs absolute risk at the "
                                 "stratification horizon (not available for this head)")
        else:
            vals_test = test_pred["risk_score"]
        groups = assign_groups(vals_test, th["thresholds"])
        names = group_names(len(th["thresholds"]) + 1)
        abs_test = test_pred["abs_risk"][:, j] if j >= 0 else None
        abs_test = abs_test if abs_test is not None and np.isfinite(abs_test).all() else None
        summ = summarise_groups(test_pred["event_time_h"], test_pred["event_indicator_h"],
                                groups, abs_risk=abs_test, horizon=self.strat_horizon,
                                names=names)
        summ["thresholds"] = th
        summ["horizon_days"] = self.strat_horizon
        return groups, names, summ

    @torch.no_grad()
    def test(self):
        # Always evaluate the saved best weights, never whatever happens to be in
        # memory. Required when --is_training 0: without this the model is random.
        self.load_checkpoint()
        if self.args.head == "cox":
            self._load_or_fit_baseline()

        out_dir = os.path.join(self.args.results, self.args.setting)
        os.makedirs(out_dir, exist_ok=True)

        _, val_loader = self._get_data("val")
        _, test_loader = self._get_data("test")
        _, train_loader = self._get_data("train", deterministic=True)
        train_pred = self.predict(train_loader)   # censoring distribution for IPCW
        val_pred = self.predict(val_loader)
        test_pred = self.predict(test_loader)

        metrics = self.evaluate(test_pred, train_pred=train_pred)
        metrics.update(self._bootstrap(test_pred))

        # calibration at the stratification horizon
        j = self.eval_horizons.index(self.strat_horizon) if self.strat_horizon in self.eval_horizons else -1
        if j >= 0 and np.isfinite(test_pred["abs_risk"][:, j]).all():
            cal_rows, ece = calibration_table(
                test_pred["event_time_h"], test_pred["event_indicator_h"],
                test_pred["abs_risk"][:, j], self.strat_horizon,
                n_bins=int(getattr(self.args, "calibration_bins", 10)))
            metrics[f"ece_{_hkey(self.strat_horizon)}"] = round(ece, 4)
            self._write_csv(os.path.join(out_dir, "calibration.csv"), cal_rows)

        # risk stratification (cut-points from validation only)
        groups, names, summ = self.stratify(val_pred, test_pred)
        metrics["risk_strat_method"] = summ["thresholds"]["method"]
        metrics["logrank_p"] = summ["logrank_p"]
        metrics["hr_high_vs_low"] = round(summ["hr_high_vs_low"], 4)
        metrics["hr_high_vs_low_ci95"] = [round(summ["hr_lo"], 4), round(summ["hr_hi"], 4)]
        for g in summ["groups"]:
            metrics[f"observed_risk_{g['group']}"] = round(g["observed_risk_km"], 4)
            metrics[f"n_{g['group']}"] = g["n"]

        metrics["checkpoint"] = self.checkpoint_file()
        metrics["setting"] = self.args.setting
        metrics["head"] = self.args.head
        metrics["eval_horizons_days"] = self.eval_horizons

        # ---- write everything ------------------------------------------------
        json.dump(metrics, open(os.path.join(out_dir, "test_metrics.json"), "w"), indent=2)
        km = {"km_grid_days": summ.pop("km_grid_days"),
              "curves": summ.pop("km_cumulative_incidence")}
        json.dump(summ, open(os.path.join(out_dir, "risk_groups.json"), "w"), indent=2)
        km_rows = [dict({"day": d}, **{name: km["curves"][name][i] for name in km["curves"]})
                   for i, d in enumerate(km["km_grid_days"])]
        self._write_csv(os.path.join(out_dir, "km_curves.csv"), km_rows)
        self._write_predictions(os.path.join(out_dir, "test_predictions.csv"), test_pred,
                                [names[g] for g in groups])
        val_groups = assign_groups(
            val_pred["risk_score"] if summ["thresholds"]["on"] == "score"
            else val_pred["abs_risk"][:, j], summ["thresholds"]["thresholds"])
        self._write_predictions(os.path.join(out_dir, "val_predictions.csv"), val_pred,
                                [names[g] for g in val_groups])

        print(f"[test] {json.dumps({k: v for k, v in metrics.items() if k != 'checkpoint'})}")
        print(f"[test] risk groups: " + " | ".join(
            f"{g['group']}: n={g['n']} observed {self.strat_horizon:.0f}d risk "
            f"{g['observed_risk_km']:.3f}" for g in summ["groups"]))
        return metrics

    # ---- writers -----------------------------------------------------------
    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def _write_predictions(self, path, pred, group_labels):
        rows = []
        for i in range(len(pred["risk_score"])):
            r = {"subject_id": int(pred["subject_id"][i]), "study_id": int(pred["study_id"][i]),
                 "risk_score": float(pred["risk_score"][i])}
            for j, h in enumerate(self.eval_horizons):
                r[f"abs_risk_{_hkey(h)}"] = float(pred["abs_risk"][i, j])
            r["risk_group"] = group_labels[i]
            r["event_time_days"] = float(pred["event_time_days"][i])
            r["event_indicator"] = int(pred["event_indicator"][i])
            r["event_by_horizon"] = int(pred["event_indicator_h"][i])
            rows.append(r)
        self._write_csv(path, rows)
