"""Mission 1 lightweight pipeline (6 channels): forecast errors -> 4 detectors -> ESA scores.

Produces the headline result: event-wise F0.5 = 0.8532 (precision 0.9993, recall 0.5385);
best published on this subset is 0.786 (Telemanom-ESA). Saved to results/M1_result.json.

Steps, in execution order:
  1. Load the trained model (results/M1_model.pt) and compute forecast errors for train + test.
  2. Step 1 cleanup: zero errors inside all-channel telemetry gaps (+30 steps).
  3. Build each detector's input:
       A  Telemanom threshold + pruning on EWMA(5) errors     (telemanom_detector.py)
       B  multi-span robust z > threshold                       (detectors.multi_span_z)
       C  silent for >= 10 steps (count == 0)                   (detectors.gap_pred_ch)
       D  B's threshold x boost within 2 h after a priority-3 telecommand
  4. Grid search on the TRAIN period only: OR the detectors per channel, close gaps, drop
     short alarms, score with ESA's event-wise F0.5. Keep the best config.
  5. Apply that frozen config ONCE to the test period and report all five ESA aspects.

Winning config: p=0.3, zmode=local, q=0.9999 (threshold 20.21), close=60, minlen=3,
gapL=10, boost=0.6. Train F0.5 0.7748.

    python src/run_m1_lightweight.py          # most of the time is the 432-config train grid
"""
import os, sys, json, time, argparse, itertools
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MISSIONS, DATA, RESULTS
from dataset import build_arrays, timestamps
from model import GDNForecaster, RF, series_errors
from detectors import (multi_span_z, ewma, gap_pred_ch, zero_gap_margin,
                       close_runs, open_runs)
from esa_metrics import full_scoreboard, global_f05, build_labels_df, PRIMARY
from telemanom_detector import telemanom_detect

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MK = "M1"
SPANS = [1, 3, 15, 60, 240]      # EWMA spans for Detector B


def prune_per_channel(errs_s, vals, p, n_ignore):
    """Detector A on every channel.

    Input : errs_s (T, C) EWMA-smoothed errors; vals (T, C) channel values; p = pruning ratio
    Output: (T, C) uint8 alarms
    """
    T, C = errs_s.shape
    out = np.zeros((T, C), np.uint8)
    for c in range(C):
        out[:, c] = telemanom_detect(errs_s[:, c], vals[:, c], p=p, num_to_ignore=n_ignore)
    return out


def assemble(prune_P, Z, thr, gapdet, tcw, boost, close, minlen):
    """Combine the four detectors into the final per-channel alarms.

    Input : prune_P (T, C) Detector A; Z (T, C) Detector-B scores and its threshold thr;
            gapdet (T, C) Detector C; tcw (T,) True within 2 h after a telecommand;
            boost = Detector-D threshold multiplier; close / minlen = cleanup sizes
    Output: (T, C) uint8
    Example (one channel): A=[0,1,1,0,0,0], B=[0,0,0,0,1,0] -> OR=[0,1,1,0,1,0]
             close=1 -> [0,1,1,1,1,0]; minlen=3 keeps it (run of 4).
    """
    P = prune_P.copy()                                                # A
    if thr is not None:
        P = np.maximum(P, (Z > thr).astype(np.uint8))                 # B
        if boost:
            P = np.maximum(P, ((Z > thr * boost) & tcw[:, None]).astype(np.uint8))   # D
    if gapdet is not None:
        P = np.maximum(P, gapdet)                                     # C
    for c in range(P.shape[1]):
        P[:, c] = open_runs(close_runs(P[:, c], close), minlen)       # cleanup
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=2880, help="Detector-B baseline block (1 day)")
    args = ap.parse_args()
    mk = MK; m = MISSIONS[mk]; RAW, CHS = m["raw"], m["chs"]; t0 = time.time()

    # ---- 1. forecast errors ----
    Xtr, mu, sd, n_ch, n_tc = build_arrays(mk, "train")
    Xte, *_ = build_arrays(mk, "test", mu, sd)                  # test uses TRAIN mu/sd
    ftr = np.load(f"{DATA}/{mk}_train_feats.npz"); fte = np.load(f"{DATA}/{mk}_test_feats.npz")
    cnt_tr, tcdt_tr = ftr["count"], ftr["tcdt"]; cnt_te, tcdt_te = fte["count"], fte["tcdt"]
    ts_tr, ts_te = timestamps(mk, "train"), timestamps(mk, "test")

    model = GDNForecaster(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{RESULTS}/{mk}_model.pt", map_location=DEV))
    err_tr = series_errors(model, Xtr, DEV); err_te = series_errors(model, Xte, DEV)
    print(f"[{mk}] errors {time.time()-t0:.0f}s", flush=True)

    # ---- 2. zero errors in all-channel gaps + 30 steps ----
    gap_tr, gap_te = (cnt_tr == 0).all(1), (cnt_te == 0).all(1)
    ec_tr = zero_gap_margin(err_tr, gap_tr, 30); ec_te = zero_gap_margin(err_te, gap_te, 30)

    # ---- 3. detector inputs ----
    es5_tr, es5_te = ewma(ec_tr, 5), ewma(ec_te, 5)                   # A: EWMA(5) errors
    Zl_tr = multi_span_z(ec_tr, SPANS, args.block); Zl_tr[:RF] = 0    # B: local baseline
    Zl_te = multi_span_z(ec_te, SPANS, args.block); Zl_te[:RF] = 0
    # B, "global" baseline. Kept as originally run so the train-time search is reproduced
    # exactly; this call has the phantom-block issue (see detectors.global_multi_span_z).
    Zg_tr = multi_span_z(ec_tr, SPANS, len(ec_tr)); Zg_tr[:RF] = 0
    Zg_te = multi_span_z(ec_te, SPANS, len(ec_te)); Zg_te[:RF] = 0
    sl = float(np.quantile(Zl_tr, 0.999)) or 1.0; sg = float(np.quantile(Zg_tr, 0.999)) or 1.0
    ZTR = {"local": Zl_tr, "global": Zg_tr, "max": np.maximum(Zl_tr/sl, Zg_tr/sg)}
    ZTE = {"local": Zl_te, "global": Zg_te, "max": np.maximum(Zl_te/sl, Zg_te/sg)}
    tcw_tr = (tcdt_tr.min(1) <= 7200.0); tcw_te = (tcdt_te.min(1) <= 7200.0)   # D: 2 h window
    print(f"[{mk}] z built {time.time()-t0:.0f}s", flush=True)

    # Detector A for each p, computed once (it is the slow part)
    P_CACHE_TR = {}
    for p in [0.05, 0.13, 0.3]:
        P_CACHE_TR[p] = prune_per_channel(es5_tr, Xtr[:, :n_ch, 0], p, RF)
        print(f"  prune p={p} train rows={int(P_CACHE_TR[p].max(1).sum()):,} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 4. tune on TRAIN ----
    grid = dict(p=[0.05, 0.13, 0.3], zmode=["none", "local", "global", "max"],
                q=[0.9999, 0.99995, 0.99999], close=[0, 60, 240], minlen=[1, 3],
                gapL=[10], boost=[None, 0.6])
    ncfg = int(np.prod([len(v) for v in grid.values()]))
    print(f"\n[{mk}] TRAIN tuning ({ncfg} configs)", flush=True)
    ldf_tr = build_labels_df(f"{RAW}/labels.csv", f"{RAW}/anomaly_types.csv", pd.to_datetime(pd.Series(ts_tr)))
    gd_tr = {L: gap_pred_ch(cnt_tr, L) for L in grid["gapL"]}                  # C
    best = (None, -1.0); i = 0
    for p, zmode, q, close, minlen, gapL, bf in itertools.product(*grid.values()):
        i += 1
        if zmode == "none":
            Z, thr = None, None
            if bf: continue                       # boost needs a z branch to act on
        else:
            Z = ZTR[zmode]; thr = float(np.quantile(Z, q))   # one threshold over all channels
        P = assemble(P_CACHE_TR[p], Z, thr, gd_tr[gapL], tcw_tr, bf, close, minlen)
        if P.sum() == 0: continue
        r = global_f05(P, ts_tr, RAW, CHS, labels_df=ldf_tr)
        f05 = r["global_EW_F_0.50"]
        if f05 > best[1]:
            best = (dict(p=p, zmode=zmode, q=q, close=close, minlen=minlen, gapL=gapL, boost=bf), f05)
            print(f"  [{i}/{ncfg}] p={p} z={zmode:<6} q={q} close={close:>3} ml={minlen} b={bf} "
                  f"-> F0.5={f05:.4f} (P={r['global_EW_precision']:.3f} R={r['global_EW_recall']:.3f} "
                  f"aP={r['global_alarming_precision']:.3f})  *", flush=True)
        elif i % 40 == 0:
            print(f"  [{i}/{ncfg}] ... best {best[1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    cfg = best[0]
    print(f"\n>>> PRE-REGISTERED (train F0.5={best[1]:.4f}): {cfg}", flush=True)

    # ---- 5. single TEST evaluation with the frozen config ----
    P_te_prune = prune_per_channel(es5_te, Xte[:, :n_ch, 0], cfg["p"], RF)
    if cfg["zmode"] == "none":
        Z_te, thr = None, None
    else:
        # threshold value comes from TRAIN z, never refitted on test
        Z_te = ZTE[cfg["zmode"]]; thr = float(np.quantile(ZTR[cfg["zmode"]], cfg["q"]))
    gd_te = gap_pred_ch(cnt_te, cfg["gapL"])
    P_te = assemble(P_te_prune, Z_te, thr, gd_te, tcw_te, cfg["boost"], cfg["close"], cfg["minlen"])
    sb = full_scoreboard(P_te, ts_te, RAW, CHS)
    print(f"\n===== {mk} TEST — ALL ESA ASPECTS =====", flush=True)
    for k in PRIMARY:
        if k in sb: print(f"  {k:34s} {sb[k]:.4f}", flush=True)
    for k, v in sb.items():
        if k not in PRIMARY: print(f"  {k:34s} {v}", flush=True)
    json.dump({"mission": mk, "config": cfg, "train_f05": best[1], "thr_value": thr,
               "spans": SPANS, "block": args.block, "test": sb},
              open(f"{RESULTS}/{mk}_result.json", "w"), indent=1)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
