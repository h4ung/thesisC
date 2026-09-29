"""Read and preprocess MIMIC-IV-ECG waveforms into model-ready tensors.

MIMIC-IV-ECG stores 12-lead, 10-second ECGs sampled at 500 Hz in WFDB format
(.hea / .dat). For each cohort study we:

  1. Load the waveform via ``wfdb``.
  2. Reorder leads to a canonical order and handle missing leads.
  3. Replace NaNs, clip extreme values, optionally band-pass / baseline-correct.
  4. Resample 500 Hz -> ``target_fs`` (paper downsamples to 250 Hz).
  5. Z-score normalise per lead (stats can be refit on the train split).
  6. Save each study as a ``.npy`` of shape (n_leads, target_len).

The Cardioformer paper segments each ECG into non-overlapping 1-second windows;
that segmentation is done on-the-fly in the dataloader so we keep the full
10-second signal here.

Requires: ``pip install wfdb scipy``. Requires the credentialed MIMIC-IV-ECG
``files/`` waveform tree mounted at ``--ecg_root``.
"""

import argparse
import os

import numpy as np
import pandas as pd

CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]


def _bandpass(sig, fs, low=0.5, high=40.0, order=3):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig, axis=-1)


def _resample(sig, fs_in, fs_out):
    if fs_in == fs_out:
        return sig
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(sig, int(fs_out // g), int(fs_in // g), axis=-1)


def load_one(record_path, ecg_root):
    """Return (signal[n_leads, L], fs). Lead order matches CANONICAL_LEADS."""
    import wfdb
    full = os.path.join(ecg_root, record_path)
    rec = wfdb.rdrecord(full)
    sig = rec.p_signal.T  # (n_leads, L)
    names = [n.upper().replace("AVR", "aVR").replace("AVL", "aVL").replace("AVF", "aVF")
             for n in rec.sig_name]
    out = np.zeros((len(CANONICAL_LEADS), sig.shape[1]), dtype=np.float32)
    for i, lead in enumerate(CANONICAL_LEADS):
        if lead in names:
            out[i] = sig[names.index(lead)]
    return out, rec.fs


def preprocess_signal(sig, fs, target_fs=250, bandpass=True, clip_uv=5.0):
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    if bandpass:
        try:
            sig = _bandpass(sig, fs)
        except Exception:
            pass
    sig = _resample(sig, fs, target_fs)
    sig = np.clip(sig, -clip_uv, clip_uv)
    # per-lead z-score (robust to flat leads)
    mu = sig.mean(axis=-1, keepdims=True)
    sd = sig.std(axis=-1, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((sig - mu) / sd).astype(np.float32)


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    cohort = pd.read_parquet(args.cohort)
    manifest = []
    for _, r in cohort.iterrows():
        study_id = r["study_id"]
        rel = str(r["ecg_path"])
        try:
            sig, fs = load_one(rel, args.ecg_root)
            proc = preprocess_signal(sig, fs, target_fs=args.target_fs)
        except Exception as e:  # keep going; log failures
            print(f"[ecg] skip study {study_id}: {e}")
            continue
        out_file = os.path.join(args.out_dir, f"{study_id}.npy")
        np.save(out_file, proc)
        manifest.append({"study_id": study_id, "npy": out_file,
                         "n_leads": proc.shape[0], "length": proc.shape[1]})
    pd.DataFrame(manifest).to_parquet(
        os.path.join(args.out_dir, "ecg_manifest.parquet"), index=False
    )
    print(f"[ecg] processed {len(manifest)} / {len(cohort)} studies -> {args.out_dir}")


def get_parser():
    p = argparse.ArgumentParser(description="Preprocess MIMIC-IV-ECG waveforms")
    p.add_argument("--cohort", default="dataset/ckd/ckd_cohort_labels.parquet")
    p.add_argument("--ecg_root", required=True, help="root holding the ECG files/ tree")
    p.add_argument("--out_dir", default="dataset/ckd/ecg_npy")
    p.add_argument("--target_fs", type=int, default=250)
    return p


if __name__ == "__main__":
    run(get_parser().parse_args())
