"""Mission 2 lightweight pipeline (11 channels): forecast errors -> absolute z rule -> ESA score.

Produces the Mission 2 result: event-wise F0.5 = 0.9644 (precision 0.9606, recall 0.98);
best published on this subset is 0.949 (Windowed iForest). Saved to results/M2_result.json.

Mission 2 needs a different decision rule than Mission 1: Telemanom pruning, which wins on
M1, scores 0.014 here. The rule is simple:
  1. errors -> EWMA(15)
  2. robust z per channel: (error - median) / IQR, statistics taken from the same split
     (label-free), then max over channels -> one score per timestep
  3. alarm wherever z > z*. z* is picked on the TRAIN period and applied to test as an
     ABSOLUTE number (z* = 20), not re-fitted as a quantile.
span = 15 was fixed by an earlier train-only selection.

    python src/run_m2_lightweight.py          # a few minutes
"""
import os, sys, json
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MISSIONS, RESULTS
from dataset import build_arrays, timestamps
from model import GDNForecaster, RF, series_errors
from detectors import ewma
from esa_metrics import global_f05

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MK = "M2"
SPAN = 15                                        # EWMA span, fixed by train selection
ZSTARS = [3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 60]   # candidate absolute thresholds


def score(pred, ts):
    """Event-wise ESA scores of a 1-D 0/1 prediction (keys without the 'global_' prefix)."""
    m = MISSIONS[MK]
    r = global_f05(pred[:, None], ts, m["raw"], m["chs"])
    return {k.replace("global_", "", 1): v for k, v in r.items()}


def zseries(err, span):
    """One anomaly score per timestep.

    Input : err (T, C) forecast errors; span = EWMA span
    Output: (T,) max over channels of (smoothed error - median) / IQR; first RF rows = 0
    Example: a channel with median 0.10, IQR 0.02 and smoothed error 0.60 -> z = 25
    """
    es = ewma(err, span)
    med = np.median(es[RF:], 0)
    iqr = np.percentile(es[RF:], 75, 0) - np.percentile(es[RF:], 25, 0) + 1e-8
    z = ((es - med) / iqr).max(1); z[:RF] = 0
    return z


def main():
    # forecast errors with the trained model (test normalised with TRAIN mu/sd)
    Xtr, mu, sd, n_ch, n_tc = build_arrays(MK, "train")
    Xte, _, _, _, _ = build_arrays(MK, "test", mu, sd)
    model = GDNForecaster(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{RESULTS}/{MK}_model.pt", map_location=DEV))
    err_tr = series_errors(model, Xtr, DEV); err_te = series_errors(model, Xte, DEV)
    del Xtr, Xte
    ts_tr, ts_te = timestamps(MK, "train"), timestamps(MK, "test")

    z_tr, z_te = zseries(err_tr, SPAN), zseries(err_te, SPAN)

    # choose z* on TRAIN labels only
    print("=== TRAIN sweep over ABSOLUTE z* ===", flush=True)
    best = (None, -1)
    for zstar in ZSTARS:
        b = (z_tr > zstar).astype(np.uint8)
        if not b.sum(): continue
        r = score(b, ts_tr)
        f = r["EW_F_0.50"]
        print(f"  z*={zstar}: F0.5={f:.4f} (P={r['EW_precision']:.3f} R={r['EW_recall']:.3f})", flush=True)
        if f > best[1]:
            best = (zstar, f)
    zstar, trf = best
    print(f"\n>> PRE-REGISTERED: span={SPAN} z*={zstar} (train F0.5={trf:.4f})", flush=True)

    # one TEST evaluation with that z*
    b = (z_te > zstar).astype(np.uint8)
    r = score(b, ts_te)
    print(f"\n=== TEST (headline, absolute transfer) ===\n"
          f"F0.5={r['EW_F_0.50']:.4f}  P={r['EW_precision']:.4f}  R={r['EW_recall']:.4f}", flush=True)
    json.dump({"span": SPAN, "zstar": zstar, "train_f05": trf,
               "test": {k: float(v) for k, v in r.items()}},
              open(f"{RESULTS}/{MK}_result.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
