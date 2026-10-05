"""Shrink MIMIC-IV hosp/labevents to the 8 lab tests the CKD model uses.

labevents has well over 100 million rows; build_cohort.py and
extract_ehr_features.py load it into memory, which can exhaust 16-32 GB of RAM.
This script streams the file in chunks (low memory) and keeps only:

    50912 creatinine   51006 BUN          50971 potassium   50983 sodium
    50882 bicarbonate  51222 haemoglobin  50931 glucose     50862 albumin

Usage (from the repo root):
    python scripts/filter_labevents.py --labevents PATH/TO/hosp/labevents.csv.gz

Writes labevents_ckd.csv.gz next to the input. Pass that file as --labevents
to build_cohort and extract_ehr_features.
"""

import argparse
import os
import time

import pandas as pd

KEEP = {50912, 51006, 50971, 50983, 50882, 51222, 50931, 50862}
COLS = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--labevents", required=True, help="hosp/labevents.csv.gz (or .csv)")
    p.add_argument("--out", default=None, help="output file (default: labevents_ckd.csv.gz next to input)")
    p.add_argument("--chunksize", type=int, default=2_000_000,
                   help="rows per chunk; lower it (e.g. 500000) if you run out of memory")
    a = p.parse_args()

    if not os.path.exists(a.labevents):
        raise SystemExit(f"File not found: {a.labevents}\nCheck the path to your MIMIC-IV hosp folder.")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.labevents)), "labevents_ckd.csv.gz")

    t0 = time.time()
    total = kept = 0
    first = True
    for chunk in pd.read_csv(a.labevents, usecols=COLS, chunksize=a.chunksize):
        sub = chunk[chunk["itemid"].isin(KEEP)]
        sub.to_csv(out, mode="w" if first else "a", header=first, index=False,
                   compression="gzip")
        first = False
        total += len(chunk)
        kept += len(sub)
        print(f"\r  read {total:,} rows, kept {kept:,}  ({time.time() - t0:.0f} s)", end="", flush=True)
    print(f"\nDone: {out}\nKept {kept:,} of {total:,} rows ({kept / max(total, 1):.1%}).")


if __name__ == "__main__":
    main()
