"""Which events does the FINAL M1 pipeline miss, and why?

Reproduces the headline M1 pipeline, then reports per-event: caught/missed,
category, locality, duration, peak error-z at several EWMA spans, burst rows,
and value-novelty. Goal: identify the dominant failure mode before choosing
the next improvement.
"""
import os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_arrays, timestamps, MISSIONS
from model import GDNv2, RF, series_errors
from score import ewma, gap_pred
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/src")
from telemanom_prune import telemanom_binary_multichannel

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ESA = "/Users/Shreyas/Downloads/Capstone/ESA"
DATA, OUT = f"{ESA}/GDNv2/data", f"{ESA}/GDNv2/results"
mk = "M1"; m = MISSIONS[mk]; RAW, CHS = m["raw"], m["chs"]
SPAN, MARGIN, GAPL, P = 5, 30, 10, 0.3


def runs_of(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.stack([np.where(d == 1)[0], np.where(d == -1)[0]], 1)


def clean(z, gap, margin):
    o = z.copy()
    for s, e in runs_of(gap):
        o[s:min(e + margin, len(z))] = 0.0
    return o


def load_events(ts):
    lab = pd.read_csv(f"{RAW}/labels.csv", parse_dates=["StartTime", "EndTime"])
    for c in ["StartTime", "EndTime"]:
        if getattr(lab[c].dt, "tz", None) is not None:
            lab[c] = lab[c].dt.tz_localize(None)
    at = pd.read_csv(f"{RAW}/anomaly_types.csv")
    lab = lab[lab.Channel.isin(CHS)].merge(at[["ID", "Category", "Locality"]], on="ID")
    lab = lab[lab.Category.isin(["Anomaly", "Rare Event"])]
    lab = lab[(lab.StartTime >= ts.iloc[0]) & (lab.EndTime <= ts.iloc[-1])]
    t0, step = ts.iloc[0], ts.iloc[1] - ts.iloc[0]
    ev = {}
    for eid, g in lab.groupby("ID"):
        s = max(int((g.StartTime.min() - t0) / step), 0)
        e = min(int(np.ceil((g.EndTime.max() - t0) / step)) + 1, len(ts))
        ev[eid] = (s, e, g.Category.iloc[0], str(g.Locality.iloc[0]))
    return ev


def main():
    Xtr, mu, sd, n_ch, n_tc = build_arrays(mk, "train")
    Xte, _, _, _, _ = build_arrays(mk, "test", mu, sd)
    del Xtr
    model = GDNv2(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{OUT}/M1_v1.pt", map_location=DEV))
    err = series_errors(model, Xte, DEV)
    f = np.load(f"{DATA}/M1_test_feats.npz")
    cnt = f["count"]; gap = (cnt == 0).all(1)
    ts = timestamps(mk, "test")

    # final pipeline
    es = ewma(err, SPAN)
    med = np.median(es[RF:], 0); iqr = np.percentile(es[RF:], 75, 0) - np.percentile(es[RF:], 25, 0) + 1e-8
    z = ((es - med) / iqr).max(1); z[:RF] = 0
    zc = clean(z, gap, MARGIN)
    esc = es.copy()
    for s, e in runs_of(gap):
        esc[s:min(e + MARGIN, len(es))] = 0.0
    pruned = telemanom_binary_multichannel(esc, Xte[:, :n_ch, 0], p=P, num_to_ignore=RF)
    pred = (pruned | gap_pred(gap, GAPL)).astype(np.uint8)
    print(f"final pred positives: {int(pred.sum()):,} ({100*pred.mean():.3f}%)")

    # alternative error views for diagnosis
    zs = {}
    for sp in [1, 3, 15, 60, 240]:
        e2 = ewma(err, sp)
        m2 = np.median(e2[RF:], 0); i2 = np.percentile(e2[RF:], 75, 0) - np.percentile(e2[RF:], 25, 0) + 1e-8
        zz = ((e2 - m2) / i2).max(1); zz[:RF] = 0
        zs[sp] = clean(zz, gap, MARGIN)
    burst = (cnt.max(1) >= 3)

    ev = load_events(ts)
    rows = []
    for eid, (s, e, cat, loc) in ev.items():
        rows.append(dict(id=eid, cat=cat[:4], loc=loc[:4], n=e - s,
                         hit=int(pred[s:e].any()),
                         gap=int(gap[s:e].sum()), burst=int(burst[s:e].sum()),
                         **{f"z{sp}": round(float(zs[sp][s:e].max()), 1) for sp in zs}))
    df = pd.DataFrame(rows)
    print(f"\ncaught {df.hit.sum()}/{len(df)} events")
    print("\n--- MISSED events (sorted by peak z at span 3) ---")
    miss = df[df.hit == 0].sort_values("z3", ascending=False)
    print(miss.to_string(index=False))
    print("\n--- CAUGHT events ---")
    print(df[df.hit == 1].sort_values("z3", ascending=False).to_string(index=False))

    print("\n--- miss rate by category / locality / duration ---")
    df["dur"] = pd.cut(df.n, [0, 5, 60, 600, 10**9], labels=["<=5", "6-60", "61-600", ">600"])
    for k in ["cat", "loc", "dur"]:
        g = df.groupby(k, observed=True).agg(n=("hit", "size"), caught=("hit", "sum"))
        g["rate"] = (100 * g.caught / g.n).round(0)
        print(f"\nby {k}:\n{g.to_string()}")

    # background reference at each span
    print("\n--- background percentiles of cleaned z ---")
    for sp in zs:
        print(f"  span {sp:>3}: p99={np.percentile(zs[sp],99):7.1f}  p99.9={np.percentile(zs[sp],99.9):8.1f}  max={zs[sp].max():9.1f}")


if __name__ == "__main__":
    main()
