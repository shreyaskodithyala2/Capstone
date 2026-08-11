"""Assemble (T, N, F) node-feature arrays for GDNv2.

Channel node features : [normalized value, gap_flag, log-scaled time-since-last-sample]
TC node features      : [exp(-dt/tau) recency decay, 0, 0]
"""
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

ESA = "/Users/Shreyas/Downloads/Capstone/ESA"
PREP = f"{ESA}/ESA-ADB/data/preprocessed/multivariate"
DATA = f"{ESA}/GDNv2/data"

MISSIONS = {
    "M1": dict(coll="ESA-Mission1-semi-supervised", raw=f"{ESA}/ESA-Mission1",
               chs=[f"channel_{i}" for i in range(41, 47)],
               train="84_months.train.csv", test="84_months.test.csv",
               split="2007-01-01", grid=30),
    "M2": dict(coll="ESA-Mission2-semi-supervised", raw=f"{ESA}/ESA-Mission2",
               chs=[f"channel_{i}" for i in range(18, 29)],
               train="21_months.train.csv", test="21_months.test.csv",
               split="2001-10-01", grid=18),
}
TC_TAU = 1800.0   # 30-min decay for telecommand recency


def build_arrays(mk, split, mu=None, sd=None):
    m = MISSIONS[mk]
    vals = pd.read_csv(f"{PREP}/{m['coll']}/{m[split]}", usecols=m["chs"])[m["chs"]] \
             .to_numpy(np.float32)
    z = np.load(f"{DATA}/{mk}_{split}_feats.npz")
    count, tsince, tcdt = z["count"], z["tsince"], z["tcdt"]
    n_tc = tcdt.shape[1]
    if mu is None:
        mu, sd = vals.mean(0), vals.std(0) + 1e-8
    vals = (vals - mu) / sd
    T, n_ch = vals.shape
    X = np.zeros((T, n_ch + n_tc, 3), np.float32)
    X[:, :n_ch, 0] = vals
    X[:, :n_ch, 1] = (count == 0).astype(np.float32)
    X[:, :n_ch, 2] = np.log10(1.0 + tsince) / 7.0
    X[:, n_ch:, 0] = np.exp(-tcdt / TC_TAU)
    return X, mu, sd, n_ch, n_tc


def timestamps(mk, split):
    m = MISSIONS[mk]
    return pd.to_datetime(pd.read_csv(f"{PREP}/{m['coll']}/{m[split]}",
                                      usecols=["timestamp"])["timestamp"])


def gap_all_mask(mk, split):
    z = np.load(f"{DATA}/{mk}_{split}_feats.npz")
    return (z["count"] == 0).all(1)


class WindowDS(Dataset):
    """x = (N, w, F) window ending at i+w-1; y = channel values at i+w (next step)."""
    def __init__(self, X, w, n_ch, stride=1):
        self.X, self.w, self.n_ch = X, w, n_ch
        self.idx = np.arange(0, len(X) - w, stride)
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        x = np.transpose(self.X[i:i + self.w], (1, 0, 2))       # (N, w, F)
        y = self.X[i + self.w, :self.n_ch, 0]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y.copy())
