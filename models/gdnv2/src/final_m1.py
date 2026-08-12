"""FINAL M1 pipeline, cleanly pre-registered:
  family  : gap-cleaned pruning + gap-detector fusion   (fixed a priori; justified by
            label-free input statistics: 16,676 count==0 runs on test vs 3,196 on train,
            whose re-acquisition spikes poison pruning's sorted-peak logic)
  margin  : 30 steps  (train-selected in iteration 2)
  L (gap) : 10 steps  (train-selected in gap analysis)
  p       : tuned on TRAIN within family; ties -> smallest p
  span    : 5 (train-selected: span-5 prune 0.726 > span-30 prune 0.688 on train)
"""
import os, sys, json
import numpy as np, torch

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
MARGIN, GAP_L, SPAN = 30, 10, 5

def runs_of(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.stack([np.where(d==1)[0], np.where(d==-1)[0]], 1)

def clean(errs, gap_mask):
    out = errs.copy()
    for s, e in runs_of(gap_mask):
        out[s:min(e+MARGIN, len(errs))] = 0.0
    return out

def offi(pred, ts):
    return official_score(pred, ts, f"{RAW}/labels.csv", f"{RAW}/anomaly_types.csv", CHS)

def main():
    Xtr, mu, sd, n_ch, n_tc = build_arrays(mk, "train")
    Xte, _, _, _, _ = build_arrays(mk, "test", mu, sd)
    model = GDNv2(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{OUT}/M1_v1.pt", map_location=DEV))
    err_tr = series_errors(model, Xtr, DEV); err_te = series_errors(model, Xte, DEV)
    vtr, vte = Xtr[:, :n_ch, 0].copy(), Xte[:, :n_ch, 0].copy()
    del Xtr, Xte
    cnt_tr = np.load(f"{DATA}/M1_train_feats.npz")["count"]
    cnt_te = np.load(f"{DATA}/M1_test_feats.npz")["count"]
    g_tr, g_te = (cnt_tr==0).all(1), (cnt_te==0).all(1)
    ts_tr, ts_te = timestamps(mk, "train"), timestamps(mk, "test")
    es_tr, es_te = clean(ewma(err_tr, SPAN), g_tr), clean(ewma(err_te, SPAN), g_te)
    gd_tr, gd_te = gap_pred(g_tr, GAP_L), gap_pred(g_te, GAP_L)

    print("=== TRAIN: choose p within the fixed family ===", flush=True)
    best = (None, -1)
    for p in [0.05, 0.13, 0.20, 0.30]:
        b = (telemanom_binary_multichannel(es_tr, vtr, p=p, num_to_ignore=RF) | gd_tr).astype(np.uint8)
        r = offi(b, ts_tr)
        f = r["EW_F_0.50"]
        print(f"  p={p}: F0.5={f:.4f} (P={r['EW_precision']:.3f} R={r['EW_recall']:.3f})", flush=True)
        if f > best[1] + 1e-6:          # strict improvement => ties keep SMALLEST p
            best = (p, f)
    p, trf = best
    print(f"\n>> FINAL PRE-REGISTERED: clean+gap, span={SPAN}, margin={MARGIN}, L={GAP_L}, p={p} "
          f"(train F0.5={trf:.4f})", flush=True)

    b = (telemanom_binary_multichannel(es_te, vte, p=p, num_to_ignore=RF) | gd_te).astype(np.uint8)
    r = offi(b, ts_te)
    print(f"\n=== TEST (FINAL HEADLINE) ===\n"
          f"F0.5={r['EW_F_0.50']:.4f}  P={r['EW_precision']:.4f}  R={r['EW_recall']:.4f}", flush=True)
    json.dump({"family": "clean+gap", "span": SPAN, "margin": MARGIN, "gap_L": GAP_L, "p": p,
               "train_f05": trf, "test": {k: float(v) for k, v in r.items()}},
              open(f"{OUT}/M1_FINAL.json", "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
