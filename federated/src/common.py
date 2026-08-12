"""Shared utilities for the federated GDN experiments."""
import os, sys, contextlib, io
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/ESA-ADB")   # timeeval metric
from timeeval.metrics import ESAScores
from official_score import official_score

PREP = "/Users/Shreyas/Downloads/Capstone/ESA/ESA-ADB/data/preprocessed/multivariate"

MISSIONS = {
    "M1": dict(coll="ESA-Mission1-semi-supervised", chs=[f"channel_{i}" for i in range(41, 47)],
               train="84_months.train.csv", test="84_months.test.csv", split="2007-01-01",
               raw="/Users/Shreyas/Downloads/Capstone/ESA/ESA-Mission1"),
    "M2": dict(coll="ESA-Mission2-semi-supervised", chs=[f"channel_{i}" for i in range(18, 29)],
               train="21_months.train.csv", test="21_months.test.csv", split="2001-10-01",
               raw="/Users/Shreyas/Downloads/Capstone/ESA/ESA-Mission2"),
}


class WindowDS(Dataset):
    def __init__(self, arr, w, stride=1):
        self.a = arr; self.w = w
        self.idx = np.arange(0, len(arr) - w, stride)
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        return torch.from_numpy(self.a[i:i + self.w].T), torch.from_numpy(self.a[i + self.w])


def load_arr(csv, chs):
    df = pd.read_csv(csv, usecols=chs)
    return df[chs].to_numpy(np.float32)


def load_mission(key, val_frac=0.15):
    m = MISSIONS[key]
    arr = load_arr(f"{PREP}/{m['coll']}/{m['train']}", m["chs"])
    mu, sd = arr.mean(0), arr.std(0) + 1e-8
    arr = (arr - mu) / sd
    n_val = int(val_frac * len(arr))
    return dict(key=key, cfg=m, mu=mu, sd=sd, tr=arr[:-n_val], val=arr[-n_val:], N=len(m["chs"]))


def evaluate_scores(model, arr, w, device, bs=8192):
    model.eval(); N = arr.shape[1]
    dl = DataLoader(WindowDS(arr, w, stride=1), batch_size=bs, shuffle=False)
    errs = np.zeros((len(arr), N), dtype=np.float32); pos = w
    with torch.no_grad():
        for x, y in dl:
            e = (model(x.to(device)) - y.to(device)).abs().cpu().numpy()
            errs[pos:pos + len(e)] = e; pos += len(e)
    return errs


def smooth_errs(errs, k):
    if k <= 1: return errs
    ker = np.ones(k, np.float32) / k
    out = np.empty_like(errs)
    for c in range(errs.shape[1]):
        out[:, c] = np.convolve(errs[:, c], ker, mode="full")[:len(errs)]
    return out


def robust_score(errs, med, iqr, w):
    A = ((errs - med) / iqr).max(axis=1)
    A[:w] = A[w:].min()
    return A


def build_y_true(cfg):
    ldf = pd.read_csv(f"{cfg['raw']}/labels.csv", parse_dates=["StartTime", "EndTime"])
    for c in ["StartTime", "EndTime"]:
        if getattr(ldf[c].dt, "tz", None) is not None:
            ldf[c] = ldf[c].dt.tz_localize(None)
    at = pd.read_csv(f"{cfg['raw']}/anomaly_types.csv")
    ldf = ldf[ldf["Channel"].isin(cfg["chs"]) & (ldf["EndTime"] > pd.to_datetime(cfg["split"]))]
    ldf = ldf.merge(at[["ID", "Category", "Dimensionality", "Locality", "Length"]], on="ID", how="left")
    return ldf.drop(columns=["Channel"]).drop_duplicates(subset=["ID"]).reset_index(drop=True)


def esa_f05(y_true, ts_dt, binary):
    yp = np.empty((len(ts_dt), 2), dtype=object)
    yp[:, 0] = ts_dt; yp[:, 1] = binary.astype(int)
    m = ESAScores(betas=0.5, select_labels={"Category": ["Rare Event", "Anomaly"]})
    with contextlib.redirect_stdout(io.StringIO()):
        return m.score(y_true, yp)


def score_on_test_official(model, mstate, w, device, smooth=10):
    """Comparable-to-baseline scoring: GDN A(t) -> binary via a VALIDATION-tuned threshold
    (no test labels used), then the faithful official ESAScores. Returns test result dict."""
    cfg, mu, sd = mstate["cfg"], mstate["mu"], mstate["sd"]
    chs = cfg["chs"]
    labels_csv = f"{cfg['raw']}/labels.csv"; at_csv = f"{cfg['raw']}/anomaly_types.csv"

    # robust stats + scores on validation
    val_err = smooth_errs(evaluate_scores(model, mstate["val"], w, device), smooth)
    med = np.median(val_err[w:], axis=0)
    iqr = (np.percentile(val_err[w:], 75, axis=0) - np.percentile(val_err[w:], 25, axis=0)) + 1e-8
    A_val = robust_score(val_err, med, iqr, w)

    # validation timestamps = last len(val) rows of the train CSV
    n_val = len(mstate["val"])
    tr_ts = pd.to_datetime(pd.read_csv(f"{PREP}/{cfg['coll']}/{cfg['train']}", usecols=["timestamp"])["timestamp"])
    val_ts = tr_ts.iloc[-n_val:].reset_index(drop=True)

    # test scores + timestamps
    te = (load_arr(f"{PREP}/{cfg['coll']}/{cfg['test']}", chs) - mu) / sd
    ts = pd.to_datetime(pd.read_csv(f"{PREP}/{cfg['coll']}/{cfg['test']}", usecols=["timestamp"])["timestamp"])
    A_te = robust_score(smooth_errs(evaluate_scores(model, te, w, device), smooth), med, iqr, w)

    # pick threshold that maximizes VALIDATION official F0.5 (no test labels touched)
    best_thr, best_valf = None, -1
    for q in [0.980, 0.990, 0.995, 0.998, 0.999, 0.9995, 0.9999]:
        thr = np.quantile(A_val, q)
        r = official_score((A_val > thr).astype(int), val_ts, labels_csv, at_csv, chs)
        if r["EW_F_0.50"] > best_valf:
            best_valf, best_thr = r["EW_F_0.50"], thr
    test = official_score((A_te > best_thr).astype(int), ts, labels_csv, at_csv, chs)
    test["_val_f05"] = best_valf
    return test


def score_on_test_telemanom(model, mstate, w, device, smooth=10, p=0.13):
    """Score GDN with Telemanom-style dynamic thresholding + pruning (per channel), then
    the faithful official ESAScores. Self-thresholding (no val tuning). Comparable to baselines."""
    from telemanom_prune import telemanom_binary_multichannel
    cfg, mu, sd = mstate["cfg"], mstate["mu"], mstate["sd"]
    chs = cfg["chs"]
    labels_csv = f"{cfg['raw']}/labels.csv"; at_csv = f"{cfg['raw']}/anomaly_types.csv"
    te = (load_arr(f"{PREP}/{cfg['coll']}/{cfg['test']}", chs) - mu) / sd
    ts = pd.to_datetime(pd.read_csv(f"{PREP}/{cfg['coll']}/{cfg['test']}", usecols=["timestamp"])["timestamp"])
    errs = smooth_errs(evaluate_scores(model, te, w, device), smooth)
    binary = telemanom_binary_multichannel(errs, te, p=p, num_to_ignore=w)
    res = official_score(binary, ts, labels_csv, at_csv, chs)
    res["_positives"] = int(binary.sum())
    return res


def score_on_test(model, mstate, w, device, smooth=10):
    """Full test-set graph-deviation scoring + official metric (oracle-best F0.5 over threshold sweep)."""
    cfg, mu, sd = mstate["cfg"], mstate["mu"], mstate["sd"]
    val_err = smooth_errs(evaluate_scores(model, mstate["val"], w, device), smooth)
    med = np.median(val_err[w:], axis=0)
    iqr = (np.percentile(val_err[w:], 75, axis=0) - np.percentile(val_err[w:], 25, axis=0)) + 1e-8
    te = (load_arr(f"{PREP}/{cfg['coll']}/{cfg['test']}", cfg["chs"]) - mu) / sd
    ts = pd.to_datetime(pd.read_csv(f"{PREP}/{cfg['coll']}/{cfg['test']}", usecols=["timestamp"])["timestamp"])
    A = robust_score(smooth_errs(evaluate_scores(model, te, w, device), smooth), med, iqr, w)
    y_true = build_y_true(cfg)
    best = {"EW_F_0.50": -1}
    for thr in np.quantile(A, [0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999]):
        r = esa_f05(y_true, ts, (A > thr))
        if r["EW_F_0.50"] > best["EW_F_0.50"]:
            best = r
    return best
