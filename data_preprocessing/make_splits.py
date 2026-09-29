"""Create subject-independent train/val/test splits.

As in the Cardioformer paper, splits are *subject-independent*: all studies from
a given patient go to exactly one split (60/20/20 by default), preventing leakage
of patient-specific signal across splits. Splitting is stratified on the binary
label at the subject level so prevalence is balanced across splits.
"""

import argparse
import os

import numpy as np
import pandas as pd


def run(args):
    df = pd.read_parquet(args.dataset)
    rng = np.random.default_rng(args.seed)

    # subject-level label = 1 if any of the subject's studies is positive
    subj = df.groupby("subject_id")["label_binary"].max().reset_index()
    pos = subj[subj["label_binary"] == 1]["subject_id"].to_numpy()
    neg = subj[subj["label_binary"] == 0]["subject_id"].to_numpy()
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_ids(ids):
        n = len(ids)
        n_tr = int(n * args.train_frac)
        n_va = int(n * args.val_frac)
        return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]

    tr_p, va_p, te_p = split_ids(pos)
    tr_n, va_n, te_n = split_ids(neg)
    assign = {}
    for sid in np.concatenate([tr_p, tr_n]):
        assign[sid] = "train"
    for sid in np.concatenate([va_p, va_n]):
        assign[sid] = "val"
    for sid in np.concatenate([te_p, te_n]):
        assign[sid] = "test"

    df["split"] = df["subject_id"].map(assign)
    out = os.path.join(args.out_dir, "ckd_dataset_split.parquet")
    df.to_parquet(out, index=False)
    print(df.groupby("split")["label_binary"].agg(["count", "mean"]))
    print(f"[split] wrote -> {out}")


def get_parser():
    p = argparse.ArgumentParser(description="Subject-independent splits")
    p.add_argument("--dataset", default="dataset/ckd/ckd_dataset.parquet")
    p.add_argument("--out_dir", default="dataset/ckd")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=41)
    return p


if __name__ == "__main__":
    run(get_parser().parse_args())
