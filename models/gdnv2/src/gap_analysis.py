"""Event-level gap analysis for M1.

Questions answered (train and test separately, but detector TUNED ON TRAIN ONLY):
  1. How many labeled events overlap a count==0 run?         (recall ceiling of a gap detector)
  2. How many count==0 runs fall outside any event?           (false-positive cost)
  3. Best min-run-length L, chosen by official F0.5 on TRAIN.
  4. Official F0.5 of the tuned gap detector alone on TEST.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/src")
from official_score import official_score

ESA = "/Users/Shreyas/Downloads/Capstone/ESA"
PREP = f"{ESA}/ESA-ADB/data/preprocessed/multivariate/ESA-Mission1-semi-supervised"
DATA = f"{ESA}/GDNv2/data"
RAW = f"{ESA}/ESA-Mission1"
CHS = [f"channel_{i}" for i in range(41, 47)]

def runs_of(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.stack([np.where(d == 1)[0], np.where(d == -1)[0]], 1)  # [start, end) index pairs

def load_events(ts):
    lab = pd.read_csv(f"{RAW}/labels.csv", parse_dates=["StartTime", "EndTime"])
    for c in ["StartTime", "EndTime"]:
        if getattr(lab[c].dt, "tz", None) is not None:
            lab[c] = lab[c].dt.tz_localize(None)
    at = pd.read_csv(f"{RAW}/anomaly_types.csv")
    lab = lab[lab.Channel.isin(CHS)].merge(at[["ID", "Category"]], on="ID")
    lab = lab[lab.Category.isin(["Anomaly", "Rare Event"])]
    lab = lab[(lab.StartTime >= ts.iloc[0]) & (lab.EndTime <= ts.iloc[-1])]
    return lab

def event_intervals(lab, ts):
    """per event ID: [min_start_idx, max_end_idx) on the grid"""
    t0, step = ts.iloc[0], (ts.iloc[1] - ts.iloc[0])
    out = {}
    for eid, g in lab.groupby("ID"):
        s = int((g.StartTime.min() - t0) / step)
        e = int(np.ceil((g.EndTime.max() - t0) / step)) + 1
        out[eid] = (max(s, 0), min(e, len(ts)), g.Category.iloc[0])
    return out

def analyze(split, csv, tune_L=None):
    ts = pd.to_datetime(pd.read_csv(f"{PREP}/{csv}", usecols=["timestamp"])["timestamp"])
    z = np.load(f"{DATA}/M1_{split}_feats.npz")
    gap_all = (z["count"] == 0).all(1)     # all 6 channels silent
    lab = load_events(ts)
    ev = event_intervals(lab, ts)
    runs = runs_of(gap_all)
    lens = runs[:, 1] - runs[:, 0]
    print(f"\n===== {split.upper()} ({len(ts):,} rows) =====")
    print(f"gap runs (all-ch silent): {len(runs)}  | len: med={np.median(lens):.0f} p90={np.percentile(lens,90):.0f} max={lens.max()}")
    print(f"events in period: {len(ev)}  ({sum(1 for v in ev.values() if v[2]=='Anomaly')} anomaly / "
          f"{sum(1 for v in ev.values() if v[2]=='Rare Event')} rare)")

    # per-event: overlapped by a gap run?
    inev = np.zeros(len(ts), bool)
    for s, e, _ in ev.values():
        inev[s:e] = True
    hit_events = sum(1 for s, e, _ in ev.values() if gap_all[s:e].any())
    out_runs = [(s, e) for s, e in runs if not inev[s:e].any()]
    print(f"events containing a gap: {hit_events}/{len(ev)} ({100*hit_events/len(ev):.0f}%)")
    print(f"gap runs fully OUTSIDE events: {len(out_runs)}/{len(runs)}  "
          f"| their lens: {sorted(e-s for s,e in out_runs)[-8:] if out_runs else []}")

    # sweep min run length L -> official F0.5 on this split
    results = {}
    Ls = [tune_L] if tune_L else [1, 2, 3, 5, 10, 20, 40, 80]
    for L in Ls:
        pred = np.zeros(len(ts), np.uint8)
        for s, e in runs[lens >= L]:
            pred[s:e] = 1
        if pred.sum() == 0:
            continue
        r = official_score(pred, ts, f"{RAW}/labels.csv", f"{RAW}/anomaly_types.csv", CHS)
        results[L] = r
        print(f"  L>={L:>3}: EW_F0.5={r['EW_F_0.50']:.4f}  (P={r['EW_precision']:.3f} R={r['EW_recall']:.3f})")
    return results

if __name__ == "__main__":
    tr = analyze("train", "84_months.train.csv")
    bestL = max(tr, key=lambda L: tr[L]["EW_F_0.50"])
    print(f"\n>> best L on TRAIN = {bestL} (F0.5={tr[bestL]['EW_F_0.50']:.4f}) -> evaluating ONLY this on TEST")
    te = analyze("test", "84_months.test.csv", tune_L=bestL)
