"""M2 iteration: extended threshold grid (down to q0.90) x smoothing span sweep.
Selection on TRAIN; chosen (span, threshold) applied once to TEST."""
import os, sys, json
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_arrays, timestamps, MISSIONS
from model import GDNv2, RF, series_errors
from score import ewma
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/src")
from official_score import official_score

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = "/Users/Shreyas/Downloads/Capstone/ESA/GDNv2/results"
mk = "M2"; m = MISSIONS[mk]; RAW, CHS = m["raw"], m["chs"]

def offi(pred, ts):
    return official_score(pred, ts, f"{RAW}/labels.csv", f"{RAW}/anomaly_types.csv", CHS)

def main():
    Xtr, mu, sd, n_ch, n_tc = build_arrays(mk, "train")
    Xte, _, _, _, _ = build_arrays(mk, "test", mu, sd)
    model = GDNv2(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{OUT}/M2_v1.pt", map_location=DEV))
    err_tr = series_errors(model, Xtr, DEV); err_te = series_errors(model, Xte, DEV)
    del Xtr, Xte
    ts_tr, ts_te = timestamps(mk, "train"), timestamps(mk, "test")

    qs = [0.90, 0.95, 0.97, 0.98, 0.985, 0.99, 0.993, 0.995, 0.997, 0.998]
    print("=== TRAIN selection ===", flush=True)
    best = (None, -1)
    for span in [5, 15, 30]:
        es_tr = ewma(err_tr, span)
        med = np.median(es_tr[RF:], 0)
        iqr = np.percentile(es_tr[RF:], 75, 0) - np.percentile(es_tr[RF:], 25, 0) + 1e-8
        z_tr = ((es_tr - med) / iqr).max(1); z_tr[:RF] = 0
        for q in qs:
            thr = float(np.quantile(z_tr, q))
            b = (z_tr > thr).astype(np.uint8)
            if not b.sum(): continue
            r = offi(b, ts_tr)
            f = r["EW_F_0.50"]
            if f > best[1]:
                best = ((span, q, thr), f)
                print(f"  span{span} q{q}: F0.5={f:.4f} (P={r['EW_precision']:.3f} R={r['EW_recall']:.3f})  <-- best", flush=True)
    (span, q, _), trf = best
    print(f"\n>> PRE-REGISTERED: span={span} q={q} (train F0.5={trf:.4f})", flush=True)

    es_te = ewma(err_te, span)
    med = np.median(es_te[RF:], 0)
    iqr = np.percentile(es_te[RF:], 75, 0) - np.percentile(es_te[RF:], 25, 0) + 1e-8
    z_te = ((es_te - med) / iqr).max(1); z_te[:RF] = 0
    thr_te = float(np.quantile(z_te, q))          # same QUANTILE rule, series-own scale
    b = (z_te > thr_te).astype(np.uint8)
    r = offi(b, ts_te)
    print(f"\n=== TEST (headline) ===\nF0.5={r['EW_F_0.50']:.4f}  P={r['EW_precision']:.4f}  R={r['EW_recall']:.4f}", flush=True)

    print("\n=== TEST analysis grid (same span) ===", flush=True)
    for q2 in qs:
        b2 = (z_te > float(np.quantile(z_te, q2))).astype(np.uint8)
        if not b2.sum(): continue
        r2 = offi(b2, ts_te)
        print(f"  q{q2}: F0.5={r2['EW_F_0.50']:.4f} (P={r2['EW_precision']:.3f} R={r2['EW_recall']:.3f})", flush=True)
    json.dump({"selected": {"span": span, "q": q}, "train_f05": trf,
               "test": {k: float(v) for k, v in r.items()}},
              open(f"{OUT}/M2_v2.json", "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
