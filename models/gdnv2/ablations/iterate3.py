"""Iteration 3 (M1): Telemanom pruning on LIGHTLY-smoothed errors (span 5, not 30 --
span-30 diluted the 1-minute anomalies), with optional gap-margin cleaning, fused with
the gap detector. Selection on TRAIN only; chosen mode applied once to TEST."""
import os, sys, json
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_arrays, timestamps, MISSIONS
from model import GDNv2, RF, series_errors
from score import ewma, gap_pred
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/src")
from official_score import official_score
from telemanom_prune import telemanom_binary_multichannel

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ESA = "/Users/Shreyas/Downloads/Capstone/ESA"
DATA = f"{ESA}/GDNv2/data"; OUT = f"{ESA}/GDNv2/results"
mk = "M1"; m = MISSIONS[mk]; RAW, CHS = m["raw"], m["chs"]

def runs_of(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.stack([np.where(d==1)[0], np.where(d==-1)[0]], 1)

def clean_errs(errs, gap_mask, margin):
    out = errs.copy()
    for s, e in runs_of(gap_mask):
        out[s:min(e+margin, len(errs))] = 0.0
    return out

def offi(pred, ts):
    return official_score(pred, ts, f"{RAW}/labels.csv", f"{RAW}/anomaly_types.csv", CHS)

def main():
    Xtr, mu, sd, n_ch, n_tc = build_arrays(mk, "train")
    Xte, _, _, _, _ = build_arrays(mk, "test", mu, sd)
    model = GDNv2(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{OUT}/M1_v1.pt", map_location=DEV))
    err_tr = series_errors(model, Xtr, DEV); err_te = series_errors(model, Xte, DEV)
    vals_tr, vals_te = Xtr[:, :n_ch, 0].copy(), Xte[:, :n_ch, 0].copy()
    del Xtr, Xte
    cnt_tr = np.load(f"{DATA}/M1_train_feats.npz")["count"]
    cnt_te = np.load(f"{DATA}/M1_test_feats.npz")["count"]
    gap_tr, gap_te = (cnt_tr==0).all(1), (cnt_te==0).all(1)
    gd_tr, gd_te = gap_pred(gap_tr, 10), gap_pred(gap_te, 10)
    ts_tr, ts_te = timestamps(mk, "train"), timestamps(mk, "test")

    es_tr, es_te = ewma(err_tr, 5), ewma(err_te, 5)

    cands = {}
    for p in [0.05, 0.13, 0.20, 0.30]:
        cands[f"p{p}"] = ("raw", p)
        cands[f"p{p}_clean"] = ("clean", p)

    print("=== TRAIN selection ===", flush=True)
    best = (None, -1)
    tr_bins = {}
    for name, (kind, p) in cands.items():
        E = clean_errs(es_tr, gap_tr, 30) if kind == "clean" else es_tr
        b = telemanom_binary_multichannel(E, vals_tr, p=p, num_to_ignore=RF)
        for suffix, bb in [("", b), ("+gap", (b | gd_tr).astype(np.uint8))]:
            if bb.sum() == 0: continue
            r = offi(bb, ts_tr)
            f = r["EW_F_0.50"]
            tr_bins[name+suffix] = (kind, p, suffix)
            print(f"  {name+suffix:>16}: F0.5={f:.4f} (P={r['EW_precision']:.3f} R={r['EW_recall']:.3f})", flush=True)
            if f > best[1]:
                best = (name+suffix, f)
    sel = best[0]; kind, p, suffix = tr_bins[sel]
    print(f"\n>> PRE-REGISTERED: {sel} (train F0.5={best[1]:.4f})", flush=True)

    E = clean_errs(es_te, gap_te, 30) if kind == "clean" else es_te
    b = telemanom_binary_multichannel(E, vals_te, p=p, num_to_ignore=RF)
    if suffix == "+gap":
        b = (b | gd_te).astype(np.uint8)
    r = offi(b, ts_te)
    print(f"\n=== TEST (headline) ===\nF0.5={r['EW_F_0.50']:.4f}  P={r['EW_precision']:.4f}  R={r['EW_recall']:.4f}", flush=True)

    print("\n=== TEST analysis grid ===", flush=True)
    for name, (kind2, p2) in cands.items():
        E2 = clean_errs(es_te, gap_te, 30) if kind2 == "clean" else es_te
        b2 = telemanom_binary_multichannel(E2, vals_te, p=p2, num_to_ignore=RF)
        for sfx, bb in [("", b2), ("+gap", (b2 | gd_te).astype(np.uint8))]:
            if bb.sum() == 0: continue
            r2 = offi(bb, ts_te)
            print(f"  {name+sfx:>16}: F0.5={r2['EW_F_0.50']:.4f} (P={r2['EW_precision']:.3f} R={r2['EW_recall']:.3f})", flush=True)
    json.dump({"selected": sel, "train_f05": best[1], "test": {k: float(v) for k, v in r.items()}},
              open(f"{OUT}/M1_v3.json", "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
