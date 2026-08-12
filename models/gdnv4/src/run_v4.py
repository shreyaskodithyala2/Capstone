"""GDNv4 = GDNv2's proven core + GDNv3's per-channel/alarming-precision fixes.

GDNv3 (threshold-only) plateaued at train F0.5 0.474 vs GDNv2's 0.726 because it
dropped Telemanom pruning -- the component that drove precision to 0.999. v4 keeps
pruning as the primary detector and layers the v3 improvements on top:

  detector branches (per channel, OR-ed):
    A. Telemanom dynamic threshold + pruning on gap-cleaned EWMA errors   [GDNv2 core]
    B. multi-span locally/globally normalized z over a train-tuned threshold [v3]
    C. per-channel gap detector (count==0 runs >= L)                      [GDNv2 core]
    D. telecommand-gated lower threshold within 2h of a priority-3 firing  [v3]
  post-processing:
    closing (merge fragments -> alarming precision) + minlen (drop blips)  [v3]

Everything tuned on TRAIN; single pre-registered TEST evaluation; all 5 ESA aspects.
"""
import os, sys, json, time, argparse, itertools
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/GDNv2/src")
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/src")
from data import build_arrays, timestamps, MISSIONS
from model import GDNv2, RF, series_errors
from detect import (multi_span_z, ewma, gap_pred_ch, zero_gap_margin,
                    close_runs, open_runs, runs_of)
from esa_metrics import full_scoreboard, global_f05, build_labels_df, PRIMARY
from telemanom_prune import telemanom_detect

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ESA = "/Users/Shreyas/Downloads/Capstone/ESA"
V2, OUT = f"{ESA}/GDNv2", f"{ESA}/GDNv3/results"
SPANS = [1, 3, 15, 60, 240]


def prune_per_channel(errs_s, vals, p, n_ignore):
    """Telemanom detect per channel -> (T, C) uint8."""
    T, C = errs_s.shape
    out = np.zeros((T, C), np.uint8)
    for c in range(C):
        out[:, c] = telemanom_detect(errs_s[:, c], vals[:, c], p=p, num_to_ignore=n_ignore)
    return out


def assemble(prune_P, Z, thr, gapdet, tcw, boost, close, minlen):
    P = prune_P.copy()
    if thr is not None:
        P = np.maximum(P, (Z > thr).astype(np.uint8))
        if boost:
            P = np.maximum(P, ((Z > thr * boost) & tcw[:, None]).astype(np.uint8))
    if gapdet is not None:
        P = np.maximum(P, gapdet)
    for c in range(P.shape[1]):
        P[:, c] = open_runs(close_runs(P[:, c], close), minlen)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="M1")
    ap.add_argument("--block", type=int, default=2880)
    ap.add_argument("--tag", default="v4")
    args = ap.parse_args()
    mk = args.mission; m = MISSIONS[mk]; RAW, CHS = m["raw"], m["chs"]; t0 = time.time()

    Xtr, mu, sd, n_ch, n_tc = build_arrays(mk, "train")
    Xte, *_ = build_arrays(mk, "test", mu, sd)
    ftr = np.load(f"{V2}/data/{mk}_train_feats.npz"); fte = np.load(f"{V2}/data/{mk}_test_feats.npz")
    cnt_tr, tcdt_tr = ftr["count"], ftr["tcdt"]; cnt_te, tcdt_te = fte["count"], fte["tcdt"]
    ts_tr, ts_te = timestamps(mk, "train"), timestamps(mk, "test")

    model = GDNv2(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{V2}/results/{mk}_v1.pt", map_location=DEV))
    err_tr = series_errors(model, Xtr, DEV); err_te = series_errors(model, Xte, DEV)
    print(f"[{mk}] errors {time.time()-t0:.0f}s", flush=True)

    gap_tr, gap_te = (cnt_tr == 0).all(1), (cnt_te == 0).all(1)
    ec_tr = zero_gap_margin(err_tr, gap_tr, 30); ec_te = zero_gap_margin(err_te, gap_te, 30)

    # branch A inputs: GDNv2 used a single EWMA span for pruning
    es5_tr, es5_te = ewma(ec_tr, 5), ewma(ec_te, 5)
    # branch B inputs
    Zl_tr = multi_span_z(ec_tr, SPANS, args.block); Zl_tr[:RF] = 0
    Zl_te = multi_span_z(ec_te, SPANS, args.block); Zl_te[:RF] = 0
    Zg_tr = multi_span_z(ec_tr, SPANS, len(ec_tr)); Zg_tr[:RF] = 0
    Zg_te = multi_span_z(ec_te, SPANS, len(ec_te)); Zg_te[:RF] = 0
    sl = float(np.quantile(Zl_tr, 0.999)) or 1.0; sg = float(np.quantile(Zg_tr, 0.999)) or 1.0
    ZTR = {"local": Zl_tr, "global": Zg_tr, "max": np.maximum(Zl_tr/sl, Zg_tr/sg)}
    ZTE = {"local": Zl_te, "global": Zg_te, "max": np.maximum(Zl_te/sl, Zg_te/sg)}
    tcw_tr = (tcdt_tr.min(1) <= 7200.0); tcw_te = (tcdt_te.min(1) <= 7200.0)
    print(f"[{mk}] z built {time.time()-t0:.0f}s", flush=True)

    # cache pruning per p (expensive)
    P_CACHE_TR, P_CACHE_TE = {}, {}
    for p in [0.05, 0.13, 0.3]:
        P_CACHE_TR[p] = prune_per_channel(es5_tr, Xtr[:, :n_ch, 0], p, RF)
        print(f"  prune p={p} train rows={int(P_CACHE_TR[p].max(1).sum()):,} ({time.time()-t0:.0f}s)", flush=True)

    grid = dict(p=[0.05, 0.13, 0.3], zmode=["none", "local", "global", "max"],
                q=[0.9999, 0.99995, 0.99999], close=[0, 60, 240], minlen=[1, 3],
                gapL=[10], boost=[None, 0.6])
    ncfg = int(np.prod([len(v) for v in grid.values()]))
    print(f"\n[{mk}] TRAIN tuning ({ncfg} configs)", flush=True)
    ldf_tr = build_labels_df(f"{RAW}/labels.csv", f"{RAW}/anomaly_types.csv", pd.to_datetime(pd.Series(ts_tr)))
    gd_tr = {L: (gap_pred_ch(cnt_tr, L) if mk == "M1" else None) for L in grid["gapL"]}
    best = (None, -1.0); i = 0
    for p, zmode, q, close, minlen, gapL, bf in itertools.product(*grid.values()):
        i += 1
        if zmode == "none":
            Z, thr = None, None
            if bf: continue                       # boost is meaningless without a z branch
        else:
            Z = ZTR[zmode]; thr = float(np.quantile(Z, q))
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

    # ---- single test evaluation ----
    P_te_prune = prune_per_channel(es5_te, Xte[:, :n_ch, 0], cfg["p"], RF)
    if cfg["zmode"] == "none":
        Z_te, thr = None, None
    else:
        Z_te = ZTE[cfg["zmode"]]; thr = float(np.quantile(ZTR[cfg["zmode"]], cfg["q"]))
    gd_te = gap_pred_ch(cnt_te, cfg["gapL"]) if mk == "M1" else None
    P_te = assemble(P_te_prune, Z_te, thr, gd_te, tcw_te, cfg["boost"], cfg["close"], cfg["minlen"])
    sb = full_scoreboard(P_te, ts_te, RAW, CHS)
    print(f"\n===== {mk} TEST — ALL ESA ASPECTS =====", flush=True)
    for k in PRIMARY:
        if k in sb: print(f"  {k:34s} {sb[k]:.4f}", flush=True)
    for k, v in sb.items():
        if k not in PRIMARY: print(f"  {k:34s} {v}", flush=True)
    json.dump({"mission": mk, "config": cfg, "train_f05": best[1], "thr_value": thr,
               "spans": SPANS, "block": args.block, "test": sb},
              open(f"{OUT}/{mk}_{args.tag}.json", "w"), indent=1)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
