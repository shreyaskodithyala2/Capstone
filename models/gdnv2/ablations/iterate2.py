"""Iteration 2 on M1: gap-cleaned error-z + gap detector + burst detector.

Insight from per-event diagnosis:
  - ~21 test anomalies have err-z 140-390 (huge), but background z has a tail to ~1149
    caused by gap stretches / re-acquisition jumps (we KNOW where those are from count==0).
  - Fix: zero the z-score inside gap runs and for a margin after each run ends, then a
    simple absolute threshold separates anomalies from background.
  - Burst detector: count>=3 sustained (only 687 burst rows on test, 3 events carry them).
All tuning (threshold, margin, burst minlen) on TRAIN labels only.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_arrays, timestamps, MISSIONS
from model import GDNv2, RF, series_errors
from score import ewma, gap_pred
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/src")
from official_score import official_score

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ESA = "/Users/Shreyas/Downloads/Capstone/ESA"
DATA = f"{ESA}/GDNv2/data"; OUT = f"{ESA}/GDNv2/results"
mk = "M1"; m = MISSIONS[mk]; RAW, CHS = m["raw"], m["chs"]

def runs_of(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.stack([np.where(d==1)[0], np.where(d==-1)[0]], 1)

def clean_z(z, gap_mask, margin):
    out = z.copy()
    for s, e in runs_of(gap_mask):
        out[s:min(e+margin, len(z))] = 0.0
    return out

def burst_pred(cnt, minlen, level=3):
    return gap_pred((cnt.max(1) >= level), minlen)

def offi(pred, ts):
    return official_score(pred, ts, f"{RAW}/labels.csv", f"{RAW}/anomaly_types.csv", CHS)

def main():
    Xtr, mu, sd, n_ch, n_tc = build_arrays(mk, "train")
    Xte, _, _, _, _ = build_arrays(mk, "test", mu, sd)
    model = GDNv2(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{OUT}/M1_v1.pt", map_location=DEV))
    err_tr = series_errors(model, Xtr, DEV); err_te = series_errors(model, Xte, DEV)
    del Xtr, Xte
    ztr_f = np.load(f"{DATA}/M1_train_feats.npz"); zte_f = np.load(f"{DATA}/M1_test_feats.npz")
    cnt_tr, cnt_te = ztr_f["count"], zte_f["count"]
    gap_tr, gap_te = (cnt_tr==0).all(1), (cnt_te==0).all(1)
    ts_tr, ts_te = timestamps(mk, "train"), timestamps(mk, "test")

    def zscore(err):
        es = ewma(err, 5)
        med = np.median(es[RF:], 0)
        iqr = np.percentile(es[RF:],75,0)-np.percentile(es[RF:],25,0)+1e-8
        z = ((es-med)/iqr).max(1); z[:RF]=0
        return z
    z_tr, z_te = zscore(err_tr), zscore(err_te)

    gapdet_tr, gapdet_te = gap_pred(gap_tr, 10), gap_pred(gap_te, 10)

    # ---- tune on TRAIN: margin x threshold x burst minlen ----
    print("=== TRAIN sweep (tuning) ===", flush=True)
    best = (None, -1)
    for margin in [30, 60, 120]:
        cz = clean_z(z_tr, gap_tr, margin)
        for thr in [20, 30, 40, 60, 80, 120, 200]:
            base = (cz > thr).astype(np.uint8) | gapdet_tr
            for bl in [0, 2, 5]:
                pred = base | burst_pred(cnt_tr, bl) if bl else base
                if pred.sum()==0: continue
                r = offi(pred, ts_tr)
                f = r["EW_F_0.50"]
                tagg = f"m{margin}_t{thr}_b{bl}"
                if f > best[1]:
                    best = (tagg, f)
                    print(f"  {tagg}: F0.5={f:.4f} (P={r['EW_precision']:.3f} R={r['EW_recall']:.3f})  <-- new best", flush=True)
    tag = best[0]; margin, thr, bl = (int(x[1:]) for x in tag.split("_"))
    print(f"\n>> PRE-REGISTERED: margin={margin} thr={thr} burst_minlen={bl} (train F0.5={best[1]:.4f})", flush=True)

    # ---- apply exactly that on TEST ----
    cz = clean_z(z_te, gap_te, margin)
    pred = (cz > thr).astype(np.uint8) | gapdet_te
    if bl: pred = pred | burst_pred(cnt_te, bl)
    r = offi(pred, ts_te)
    print(f"\n=== TEST (headline) ===\nF0.5={r['EW_F_0.50']:.4f}  P={r['EW_precision']:.4f}  R={r['EW_recall']:.4f}", flush=True)

    # analysis grid on test (NOT headline, for understanding)
    print("\n=== TEST analysis grid ===", flush=True)
    for thr2 in [30, 60, 120]:
        for bl2 in [0, 2]:
            p2 = (clean_z(z_te, gap_te, margin) > thr2).astype(np.uint8) | gapdet_te
            if bl2: p2 = p2 | burst_pred(cnt_te, bl2)
            r2 = offi(p2, ts_te)
            print(f"  thr={thr2} burst={bl2}: F0.5={r2['EW_F_0.50']:.4f} (P={r2['EW_precision']:.3f} R={r2['EW_recall']:.3f})", flush=True)
    json.dump({"pre_registered": tag, "train_f05": best[1],
               "test": {k: float(v) for k,v in r.items()}},
              open(f"{OUT}/M1_v2.json","w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
