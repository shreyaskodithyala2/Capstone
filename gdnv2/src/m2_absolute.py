"""M2: absolute-z threshold transfer (design-consistent with robust normalization).
Train picks (span, z*) by F0.5; z* applied to test AS AN ABSOLUTE value.
Pre-committed before seeing test results for this rule."""
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

def zseries(err, span):
    es = ewma(err, span)
    med = np.median(es[RF:], 0)
    iqr = np.percentile(es[RF:], 75, 0) - np.percentile(es[RF:], 25, 0) + 1e-8
    z = ((es - med) / iqr).max(1); z[:RF] = 0
    return z

def main():
    Xtr, mu, sd, n_ch, n_tc = build_arrays(mk, "train")
    Xte, _, _, _, _ = build_arrays(mk, "test", mu, sd)
    model = GDNv2(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{OUT}/M2_v1.pt", map_location=DEV))
    err_tr = series_errors(model, Xtr, DEV); err_te = series_errors(model, Xte, DEV)
    del Xtr, Xte
    ts_tr, ts_te = timestamps(mk, "train"), timestamps(mk, "test")

    span = 15                                    # fixed by the previous train selection
    z_tr, z_te = zseries(err_tr, span), zseries(err_te, span)

    print("=== TRAIN sweep over ABSOLUTE z* ===", flush=True)
    best = (None, -1)
    for zstar in [3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 60]:
        b = (z_tr > zstar).astype(np.uint8)
        if not b.sum(): continue
        r = offi(b, ts_tr)
        f = r["EW_F_0.50"]
        print(f"  z*={zstar}: F0.5={f:.4f} (P={r['EW_precision']:.3f} R={r['EW_recall']:.3f})", flush=True)
        if f > best[1]:
            best = (zstar, f)
    zstar, trf = best
    print(f"\n>> PRE-REGISTERED: span={span} z*={zstar} (train F0.5={trf:.4f})", flush=True)

    b = (z_te > zstar).astype(np.uint8)
    r = offi(b, ts_te)
    print(f"\n=== TEST (headline, absolute transfer) ===\n"
          f"F0.5={r['EW_F_0.50']:.4f}  P={r['EW_precision']:.4f}  R={r['EW_recall']:.4f}", flush=True)
    json.dump({"span": span, "zstar": zstar, "train_f05": trf,
               "test": {k: float(v) for k, v in r.items()}},
              open(f"{OUT}/M2_v3_absolute.json", "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
