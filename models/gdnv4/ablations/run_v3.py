"""GDNv3: tune the full detection stack on TRAIN, pre-register, evaluate once on TEST.

Protocol (no cheating):
  1. Errors come from the GDNv2 model trained on the train split only.
  2. EVERY hyper-parameter is selected by official global EW-F0.5 on the TRAIN period.
  3. The winning config is printed and frozen BEFORE test scoring.
  4. TEST is scored once, reporting all five ESA aspects.
"""
import os, sys, json, time, argparse, itertools
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/GDNv2/src")
from data import build_arrays, timestamps, MISSIONS
from model import GDNv2, RF, series_errors
from detect import (multi_span_z, gap_pred_ch, zero_gap_margin, build_pred, runs_of)
from esa_metrics import full_scoreboard, global_f05, build_labels_df, PRIMARY

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ESA = "/Users/Shreyas/Downloads/Capstone/ESA"
V2, OUT = f"{ESA}/GDNv2", f"{ESA}/GDNv3/results"
SPANS = [1, 3, 15, 60, 240]


def tc_window(tcdt, hours, grid_s):
    """Boolean: within `hours` after ANY priority-3 telecommand firing."""
    return (tcdt.min(axis=1) <= hours * 3600.0)


def prep(mk, split, mu=None, sd=None):
    X, mu, sd, n_ch, n_tc = build_arrays(mk, split, mu, sd)
    f = np.load(f"{V2}/data/{mk}_{split}_feats.npz")
    return X, mu, sd, n_ch, n_tc, f["count"], f["tcdt"], timestamps(mk, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="M1")
    ap.add_argument("--block", type=int, default=2880)     # 24h at 30s
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()
    mk = args.mission; m = MISSIONS[mk]; RAW, CHS = m["raw"], m["chs"]
    grid_s = m["grid"]; t0 = time.time()

    Xtr, mu, sd, n_ch, n_tc, cnt_tr, tcdt_tr, ts_tr = prep(mk, "train")
    Xte, *_ = build_arrays(mk, "test", mu, sd)
    f_te = np.load(f"{V2}/data/{mk}_test_feats.npz")
    cnt_te, tcdt_te = f_te["count"], f_te["tcdt"]
    ts_te = timestamps(mk, "test")

    model = GDNv2(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{V2}/results/{mk}_v1.pt", map_location=DEV))
    err_tr = series_errors(model, Xtr, DEV)
    err_te = series_errors(model, Xte, DEV)
    print(f"[{mk}] errors computed  {time.time()-t0:.0f}s", flush=True)

    gap_any_tr, gap_any_te = (cnt_tr == 0).all(1), (cnt_te == 0).all(1)
    ec_tr = zero_gap_margin(err_tr, gap_any_tr, 30)
    ec_te = zero_gap_margin(err_te, gap_any_te, 30)

    # LOCAL (block) normalization lifts subtle events in quiet stretches, but can suppress
    # very long events (they become their own baseline). GLOBAL keeps those. Train picks.
    Zl_tr = multi_span_z(ec_tr, SPANS, args.block); Zl_tr[:RF] = 0
    Zl_te = multi_span_z(ec_te, SPANS, args.block); Zl_te[:RF] = 0
    Zg_tr = multi_span_z(ec_tr, SPANS, len(ec_tr)); Zg_tr[:RF] = 0     # one block == global
    Zg_te = multi_span_z(ec_te, SPANS, len(ec_te)); Zg_te[:RF] = 0
    # put both on a comparable scale (train quantile) before combining
    sl = float(np.quantile(Zl_tr, 0.999)) or 1.0
    sg = float(np.quantile(Zg_tr, 0.999)) or 1.0
    ZTR = {"local": Zl_tr, "global": Zg_tr, "max": np.maximum(Zl_tr / sl, Zg_tr / sg)}
    ZTE = {"local": Zl_te, "global": Zg_te, "max": np.maximum(Zl_te / sl, Zg_te / sg)}
    print(f"[{mk}] z built {time.time()-t0:.0f}s | local p99={np.percentile(Zl_tr,99):.1f} "
          f"p99.9={np.percentile(Zl_tr,99.9):.1f} | global p99={np.percentile(Zg_tr,99):.1f} "
          f"p99.9={np.percentile(Zg_tr,99.9):.1f}", flush=True)

    tcw_tr = tc_window(tcdt_tr, 2.0, grid_s)
    tcw_te = tc_window(tcdt_te, 2.0, grid_s)

    # ---------------- TUNE ON TRAIN ONLY ----------------
    QS = (0.999, 0.9995, 0.9998, 0.9999, 0.99995, 0.99999)
    grid = dict(zmode=["local", "global", "max"], q=list(QS),
                close=[0, 60, 240], minlen=[1, 3], gapL=[10], boost=[None, 0.6])
    ncfg = int(np.prod([len(v) for v in grid.values()]))
    print(f"\n[{mk}] TRAIN tuning ({ncfg} configs, fast global-only scoring)", flush=True)
    ldf_tr = build_labels_df(f"{RAW}/labels.csv", f"{RAW}/anomaly_types.csv",
                             pd.to_datetime(pd.Series(ts_tr)))
    gd_cache = {L: (gap_pred_ch(cnt_tr, L) if mk == "M1" else None) for L in grid["gapL"]}
    best = (None, -1.0); i = 0
    for zmode, q, close, minlen, gapL, bf in itertools.product(*grid.values()):
        i += 1
        Z = ZTR[zmode]; thr = float(np.quantile(Z, q))
        tb, thb = (tcw_tr, thr * bf) if bf else (None, None)
        P = build_pred(Z, thr, close, minlen, gd_cache[gapL], tb, thb)
        if P.sum() == 0:
            continue
        r = global_f05(P, ts_tr, RAW, CHS, labels_df=ldf_tr)
        f05 = r["global_EW_F_0.50"]
        if f05 > best[1]:
            best = (dict(zmode=zmode, q=q, close=close, minlen=minlen, gapL=gapL, boost=bf), f05)
            print(f"  [{i}/{ncfg}] z={zmode:<6} q={q} close={close:>3} minlen={minlen} boost={bf} "
                  f"-> F0.5={f05:.4f} (P={r['global_EW_precision']:.3f} "
                  f"R={r['global_EW_recall']:.3f} alarmP={r['global_alarming_precision']:.3f})  *", flush=True)
        elif i % 25 == 0:
            print(f"  [{i}/{ncfg}] ... best so far {best[1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    cfg = best[0]
    print(f"\n>>> PRE-REGISTERED CONFIG (train F0.5={best[1]:.4f}): {cfg}", flush=True)

    # ---------------- SINGLE TEST EVALUATION ----------------
    # threshold transfers as the TRAIN-derived quantile VALUE (absolute), not a test quantile
    thr = float(np.quantile(ZTR[cfg["zmode"]], cfg["q"]))
    gd_te = gap_pred_ch(cnt_te, cfg["gapL"]) if mk == "M1" else None
    tb, thb = (tcw_te, thr * cfg["boost"]) if cfg["boost"] else (None, None)
    P_te = build_pred(ZTE[cfg["zmode"]], thr, cfg["close"], cfg["minlen"], gd_te, tb, thb)
    sb = full_scoreboard(P_te, ts_te, RAW, CHS)
    print(f"\n===== {mk} TEST — ALL ESA ASPECTS =====", flush=True)
    for k in PRIMARY:
        if k in sb:
            print(f"  {k:34s} {sb[k]:.4f}", flush=True)
    for k, v in sb.items():
        if k not in PRIMARY:
            print(f"  {k:34s} {v}", flush=True)

    json.dump({"mission": mk, "config": cfg, "train_f05": best[1], "spans": SPANS, "thr_value": thr,
               "block": args.block, "test": sb},
              open(f"{OUT}/{mk}_{args.tag}.json", "w"), indent=1)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
