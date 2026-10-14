"""Builds the model's input tensor X (T, N, 3) for one mission and one split.

The graph has N = n_ch + n_tc nodes: every telemetry channel and every priority-3
telecommand. Each node gets 3 numbers per timestep:

    channel node     : [value_z, gap_flag, log_tsince]
                        value_z    = (value - train_mean) / train_std
                        gap_flag   = 1 if no raw sample arrived in this grid slot (count == 0)
                        log_tsince = log10(1 + seconds since last raw sample) / 7   (~0..1)
    telecommand node : [recency, 0, 0]
                        recency    = exp(-seconds since the command last fired / 1800)

Sources: channel values from ESA's preprocessed CSV (or a pre-exported .npy of it),
gap_flag / tsince / recency from our *_feats.npz (made by extract_features.py).
Labels are never part of X.
"""
import os
import numpy as np, pandas as pd

from config import MISSIONS, PREP, DATA, TC_TAU


def _npy(name):
    """Path of a pre-exported array, or None. Looks in DATA, then DATA/full
    (locally the full-set arrays live in data/full/, on Kaggle they sit directly in DATA)."""
    for d in (DATA, f"{DATA}/full"):
        if os.path.exists(f"{d}/{name}"):
            return f"{d}/{name}"
    return None


def _vals(mk, split):
    """Raw channel values for one split.

    Input : mk = mission key ("M1", "M1_full", ...), split = "train" | "test".
    Output: (T, n_ch) float32, columns in MISSIONS[mk]["chs"] order.
    Uses the .npy export when it exists (same numbers, loads in seconds); otherwise parses
    the official CSV, which takes minutes for 7.36M rows.
    """
    m = MISSIONS[mk]
    npy = _npy(f"{mk}_{split}_vals.npy")
    if npy:
        return np.load(npy)
    # dtype=float32 at parse time avoids a 4.5 GB float64 frame on the full set
    return pd.read_csv(f"{PREP}/{m['coll']}/{m[split]}", usecols=m["chs"],
                       dtype=np.float32)[m["chs"]].to_numpy(np.float32)


def build_arrays(mk, split, mu=None, sd=None):
    """Assemble the model input tensor for one split.

    Input : mk, split; mu/sd = per-channel normalisation stats. Leave them None for the
            train split (they are computed from it); pass train's mu/sd for the test split.
    Output: X (T, N, 3) float32, mu (n_ch,), sd (n_ch,), n_ch, n_tc
    Example: build_arrays("M1", "train") -> X (7_364_161, 17, 3), n_ch=6, n_tc=11
    """
    m = MISSIONS[mk]
    vals = _vals(mk, split)
    z = np.load(f"{DATA}/{mk}_{split}_feats.npz")
    count, tsince, tcdt = z["count"], z["tsince"], z["tcdt"]
    n_tc = tcdt.shape[1]
    if mu is None:                                        # train split: fit the stats
        mu = vals.mean(0)
        sd = np.maximum(vals.std(0), m.get("sd_floor", 0.0)) + 1e-8
    vals = (vals - mu) / sd
    T, n_ch = vals.shape
    X = np.zeros((T, n_ch + n_tc, 3), np.float32)
    X[:, :n_ch, 0] = vals                                 # channel: normalised value
    X[:, :n_ch, 1] = (count == 0).astype(np.float32)      # channel: 1 = nothing arrived
    X[:, :n_ch, 2] = np.log10(1.0 + tsince) / 7.0         # channel: how stale the value is
    X[:, n_ch:, 0] = np.exp(-tcdt / TC_TAU)               # telecommand: 1 right after firing
    return X, mu, sd, n_ch, n_tc


def timestamps(mk, split):
    """Grid timestamps of one split as a pandas Series of datetime64 (length T)."""
    m = MISSIONS[mk]
    npy = _npy(f"{mk}_{split}_ts.npy")
    if npy:
        return pd.Series(np.load(npy).astype("datetime64[ns]"))
    return pd.to_datetime(pd.read_csv(f"{PREP}/{m['coll']}/{m[split]}",
                                      usecols=["timestamp"])["timestamp"])
