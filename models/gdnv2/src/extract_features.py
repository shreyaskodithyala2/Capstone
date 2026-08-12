"""Extract availability + telecommand features from RAW pickles, aligned to the
preprocessed grid timestamps. These are label-free and computable onboard in real time.

Per grid point t (bucket (t-grid, t], matching what ZOH could see at t):
  count(t)  = number of raw samples in the bucket           -> uint16
  tsince(t) = seconds since the last raw sample <= t        -> float32
Per telecommand: tcdt(t) = seconds since last firing <= t   -> float32 (cap 1e7)
"""
import os, sys, time
import numpy as np, pandas as pd

ESA = "/Users/Shreyas/Downloads/Capstone/ESA"
PREP = f"{ESA}/ESA-ADB/data/preprocessed/multivariate"
OUT = f"{ESA}/GDNv2/data"

MISSIONS = {
    "M1": dict(coll="ESA-Mission1-semi-supervised", raw=f"{ESA}/ESA-Mission1",
               chs=[f"channel_{i}" for i in range(41, 47)],
               train="84_months.train.csv", test="84_months.test.csv"),
    "M2": dict(coll="ESA-Mission2-semi-supervised", raw=f"{ESA}/ESA-Mission2",
               chs=[f"channel_{i}" for i in range(18, 29)],
               train="21_months.train.csv", test="21_months.test.csv"),
}

def grid_ns(csv):
    ts = pd.read_csv(csv, usecols=["timestamp"])["timestamp"]
    t = pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S")
    return t.to_numpy().astype("datetime64[ns]").astype(np.int64)

def avail_for_channel(raw_ns, g_ns, grid_step_ns):
    idx = np.searchsorted(raw_ns, g_ns, side="right")
    prev_idx = np.searchsorted(raw_ns, g_ns - grid_step_ns, side="right")
    count = (idx - prev_idx).astype(np.uint16)
    tsince = np.full(len(g_ns), 1e7, dtype=np.float32)
    has = idx > 0
    tsince[has] = (g_ns[has] - raw_ns[idx[has] - 1]) / 1e9
    return count, tsince

def main():
    for mk, m in MISSIONS.items():
        # kept telecommands = priority >= 3 (same rule as official preprocessing)
        tc_meta = pd.read_csv(f"{m['raw']}/telecommands.csv")
        tcs = sorted(tc_meta[tc_meta.Priority >= 3].Telecommand.tolist())
        tc_ns = {}
        for t in tcs:
            d = pd.read_pickle(f"{m['raw']}/telecommands/{t}.zip")
            tc_ns[t] = pd.to_datetime(d.index).to_numpy().astype("datetime64[ns]").astype(np.int64)
        raw_times = {}
        for c in m["chs"]:
            t0 = time.time()
            df = pd.read_pickle(f"{m['raw']}/channels/{c}.zip")
            raw_times[c] = df.index.to_numpy().astype("datetime64[ns]").astype(np.int64)
            del df
            print(f"{mk} {c}: {len(raw_times[c]):,} raw samples ({time.time()-t0:.0f}s)", flush=True)

        for split in ["train", "test"]:
            g = grid_ns(f"{PREP}/{m['coll']}/{m[split]}")
            step = g[1] - g[0]
            T, N = len(g), len(m["chs"])
            count = np.zeros((T, N), np.uint16)
            tsince = np.zeros((T, N), np.float32)
            for j, c in enumerate(m["chs"]):
                count[:, j], tsince[:, j] = avail_for_channel(raw_times[c], g, step)
            tcdt = np.full((T, len(tcs)), 1e7, np.float32)
            for j, t in enumerate(tcs):
                idx = np.searchsorted(tc_ns[t], g, side="right")
                has = idx > 0
                tcdt[has, j] = np.minimum((g[has] - tc_ns[t][idx[has] - 1]) / 1e9, 1e7)
            np.savez_compressed(f"{OUT}/{mk}_{split}_feats.npz",
                                count=count, tsince=tsince, tcdt=tcdt,
                                channels=np.array(m["chs"]), tcs=np.array(tcs))
            gap = (count == 0)
            print(f"{mk} {split}: T={T:,}  gap-rows(any ch)={int(gap.any(1).sum()):,} "
                  f"({100*gap.any(1).mean():.2f}%)  tc_cols={len(tcs)}", flush=True)

if __name__ == "__main__":
    main()
