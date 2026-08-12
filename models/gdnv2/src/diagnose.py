"""Per-event diagnosis on M1: for every labeled event (train + test), measure every
candidate detection signal, then tune additional detectors on TRAIN and fuse.

Signals per event:
  gap        : count==0 run >= L inside event
  burst      : count >= B sustained inside event      (burst-mode telemetry)
  value_z    : max |value robust-z vs TRAIN dist| inside event   (novelty / Global)
  err_z      : max forecast-error z inside event (span sweep)
"""
import os, sys, json
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_arrays, timestamps, MISSIONS
from model import GDNv2, RF, series_errors
from score import ewma, gap_pred
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/src")
from official_score import official_score

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ESA = "/Users/Shreyas/Downloads/Capstone/ESA"
DATA = f"{ESA}/GDNv2/data"
OUT = f"{ESA}/GDNv2/results"
mk = "M1"; m = MISSIONS[mk]; CHS = m["chs"]; RAW = m["raw"]

def load_events(ts):
    lab = pd.read_csv(f"{RAW}/labels.csv", parse_dates=["StartTime", "EndTime"])
    for c in ["StartTime", "EndTime"]:
        if getattr(lab[c].dt, "tz", None) is not None:
            lab[c] = lab[c].dt.tz_localize(None)
    at = pd.read_csv(f"{RAW}/anomaly_types.csv")
    lab = lab[lab.Channel.isin(CHS)].merge(at[["ID","Category","Locality","Length"]], on="ID")
    lab = lab[lab.Category.isin(["Anomaly","Rare Event"])]
    lab = lab[(lab.StartTime >= ts.iloc[0]) & (lab.EndTime <= ts.iloc[-1])]
    t0, step = ts.iloc[0], ts.iloc[1]-ts.iloc[0]
    ev = {}
    for eid, g in lab.groupby("ID"):
        s = max(int((g.StartTime.min()-t0)/step), 0)
        e = min(int(np.ceil((g.EndTime.max()-t0)/step))+1, len(ts))
        ev[eid] = dict(s=s, e=e, cat=g.Category.iloc[0], loc=g.Locality.iloc[0], length=e-s)
    return ev

def main():
    Xtr, mu, sd, n_ch, n_tc = build_arrays(mk, "train")
    Xte, _, _, _, _ = build_arrays(mk, "test", mu, sd)
    vsplit = int(0.85*len(Xtr))
    model = GDNv2(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{OUT}/M1_v1.pt", map_location=DEV))

    err_tr = series_errors(model, Xtr, DEV)
    err_te = series_errors(model, Xte, DEV)
    ztr = np.load(f"{DATA}/M1_train_feats.npz"); zte = np.load(f"{DATA}/M1_test_feats.npz")
    cnt_tr, cnt_te = ztr["count"], zte["count"]

    # value-novelty z (robust, per channel, stats from train nominal-ish whole period)
    v_med = np.median(Xtr[:, :n_ch, 0], 0)
    v_iqr = np.percentile(Xtr[:, :n_ch, 0], 75, 0) - np.percentile(Xtr[:, :n_ch, 0], 25, 0) + 1e-8
    vz_tr = np.abs((Xtr[:, :n_ch, 0]-v_med)/v_iqr).max(1)
    vz_te = np.abs((Xte[:, :n_ch, 0]-v_med)/v_iqr).max(1)

    for split, X, err, cnt, vz in [("train", Xtr, err_tr, cnt_tr, vz_tr),
                                    ("test", Xte, err_te, cnt_te, vz_te)]:
        ts = timestamps(mk, split)
        ev = load_events(ts)
        es5 = ewma(err, 5); es30 = ewma(err, 30)
        med5 = np.median(es5[vsplit:] if split=="train" else es5[RF:], 0)
        iqr5 = np.percentile(es5[vsplit:] if split=="train" else es5[RF:],75,0) - \
               np.percentile(es5[vsplit:] if split=="train" else es5[RF:],25,0) + 1e-8
        z5 = ((es5-med5)/iqr5).max(1); z5[:RF]=0
        gap = (cnt==0).all(1)
        burst = (cnt.max(1) >= 3)
        rows=[]
        for eid,e in ev.items():
            s,t = e["s"], e["e"]
            rows.append(dict(id=eid, cat=e["cat"][:4], loc=str(e["loc"])[:4], n=t-s,
                gap=int(gap[s:t].sum()), burst=int(burst[s:t].sum()),
                vz=float(vz[s:t].max()), errz=float(z5[s:t].max())))
        df = pd.DataFrame(rows).sort_values("errz")
        print(f"\n===== {split.upper()} events ({len(df)}) =====")
        print(df.to_string(index=False))
        # background reference for thresholds
        print(f"background z5: p99={np.percentile(z5,99):.1f} p999={np.percentile(z5,99.9):.1f} max={z5.max():.0f}")
        print(f"background vz: p99={np.percentile(vz,99):.1f} p999={np.percentile(vz,99.9):.1f} max={vz.max():.0f}")
        bo = burst & ~np.array([False]*len(burst))
        print(f"burst rows: {int(burst.sum())} | gap rows: {int(gap.sum())}")

if __name__ == "__main__":
    main()
