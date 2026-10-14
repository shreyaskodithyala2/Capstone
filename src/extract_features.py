"""Recover data-availability features from ESA-ADB's RAW pickles (our key preprocessing step).

ESA's official preprocessing resamples every channel onto a fixed grid with zero-order hold:
when no sample arrives, the previous value is copied forward. That makes "fresh sample" and
"nothing arrived" look identical. On Mission 1, grid slots with count == 0 are anomalous
56.7 % of the time, so this lost information matters. We rebuild it from the raw timestamps.

For every grid timestamp t (bucket = (t - grid_step, t]):
    count[t, c]  = number of raw samples of channel c in the bucket         uint16
    tsince[t, c] = seconds since channel c's last raw sample at or before t   float32
    tcdt[t, k]   = seconds since priority-3 telecommand k last fired (cap 1e7)  float32
All three are label-free and could be computed onboard in real time.

    python src/extract_features.py              # M1 + M2 lightweight
    python src/extract_features.py M1_full      # one full channel set
Output: data/<mission>_<split>_feats.npz with arrays count, tsince, tcdt, channels, tcs.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MISSIONS, PREP, DATA


def grid_ns(csv):
    """Grid timestamps of an official preprocessed CSV, as int64 nanoseconds."""
    ts = pd.read_csv(csv, usecols=["timestamp"])["timestamp"]
    t = pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S")
    return t.to_numpy().astype("datetime64[ns]").astype(np.int64)


def avail_for_channel(raw_ns, g_ns, grid_step_ns):
    """count and tsince of one channel at every grid point.

    Input : raw_ns (sorted raw sample times, ns), g_ns (grid times, ns), grid_step_ns
    Output: count (T,) uint16, tsince (T,) float32 seconds (1e7 = never seen yet)
    Example: raw samples at 0 s and 65 s, grid every 30 s at 30/60/90 s
             -> count = [0, 0, 1], tsince = [30, 60, 25]
    """
    idx = np.searchsorted(raw_ns, g_ns, side="right")                  # samples <= t
    prev_idx = np.searchsorted(raw_ns, g_ns - grid_step_ns, side="right")  # samples <= t-step
    count = (idx - prev_idx).astype(np.uint16)
    tsince = np.full(len(g_ns), 1e7, dtype=np.float32)
    has = idx > 0
    tsince[has] = (g_ns[has] - raw_ns[idx[has] - 1]) / 1e9             # t - last sample time
    return count, tsince


def main():
    keys = sys.argv[1:] or ["M1", "M2"]       # full sets only when asked for explicitly
    os.makedirs(DATA, exist_ok=True)
    for mk in keys:
        m = MISSIONS[mk]
        # keep priority >= 3 telecommands: the same rule as ESA's official preprocessing
        tc_meta = pd.read_csv(f"{m['raw']}/telecommands.csv")
        tcs = sorted(tc_meta[tc_meta.Priority >= 3].Telecommand.tolist())
        tc_ns = {}
        for t in tcs:
            d = pd.read_pickle(f"{m['raw']}/telecommands/{t}.zip")       # index = firing times
            tc_ns[t] = pd.to_datetime(d.index).to_numpy().astype("datetime64[ns]").astype(np.int64)
        # One raw channel in memory at a time (caching all 76 took 6.7 GB on M1).
        for split in ["train", "test"]:
            g = grid_ns(f"{PREP}/{m['coll']}/{m[split]}")
            step = g[1] - g[0]
            T, N = len(g), len(m["chs"])
            count = np.zeros((T, N), np.uint16)
            tsince = np.zeros((T, N), np.float32)
            for j, c in enumerate(m["chs"]):
                df = pd.read_pickle(f"{m['raw']}/channels/{c}.zip")       # index = sample times
                raw_ns = df.index.to_numpy().astype("datetime64[ns]").astype(np.int64)
                del df
                count[:, j], tsince[:, j] = avail_for_channel(raw_ns, g, step)
                del raw_ns
            tcdt = np.full((T, len(tcs)), 1e7, np.float32)
            for j, t in enumerate(tcs):
                idx = np.searchsorted(tc_ns[t], g, side="right")
                has = idx > 0
                tcdt[has, j] = np.minimum((g[has] - tc_ns[t][idx[has] - 1]) / 1e9, 1e7)
            np.savez_compressed(f"{DATA}/{mk}_{split}_feats.npz",
                                count=count, tsince=tsince, tcdt=tcdt,
                                channels=np.array(m["chs"]), tcs=np.array(tcs))
            gap = (count == 0)
            print(f"{mk} {split}: T={T:,}  gap-rows(any ch)={int(gap.any(1).sum()):,} "
                  f"({100*gap.any(1).mean():.2f}%)  tc_cols={len(tcs)}", flush=True)
            del count, tsince, tcdt, gap


if __name__ == "__main__":
    main()
