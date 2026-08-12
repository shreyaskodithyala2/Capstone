"""Scoring: forecast errors -> EWMA smoothing -> robust z -> operating points,
optionally fused (OR) with the count==0 gap detector. Official metric throughout."""
import sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/src")
from official_score import official_score
from telemanom_prune import telemanom_binary_multichannel


def forecast_errors(model, X, w, n_ch, device, bs=2048):
    from data import WindowDS
    model.eval()
    errs = np.zeros((len(X), n_ch), np.float32)
    dl = DataLoader(WindowDS(X, w, n_ch, stride=1), batch_size=bs, shuffle=False)
    pos = w
    with torch.no_grad():
        for x, y in dl:
            p = model(x.to(device))
            e = (p - y.to(device)).abs().cpu().numpy()
            errs[pos:pos + len(e)] = e
            pos += len(e)
    return errs


def ewma(errs, span):
    return pd.DataFrame(errs).ewm(span=span, adjust=False).mean().to_numpy(np.float32)


def robust_z(errs, med, iqr):
    return ((errs - med) / iqr).max(axis=1)


def gap_pred(gap_mask, min_len):
    d = np.diff(np.concatenate([[0], gap_mask.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    out = np.zeros(len(gap_mask), np.uint8)
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            out[s:e] = 1
    return out


def score_operating_points(A, ts, raw, chs, thresholds, prune_input=None, gap=None, p=0.13, w=250):
    """A: robust-z series. Returns dict name -> (binary, official result)."""
    out = {}
    for name, thr in thresholds.items():
        b = (A > thr).astype(np.uint8)
        if b.sum() == 0:
            continue
        out[name] = b
    if prune_input is not None:
        errs, vals = prune_input
        out["prune"] = telemanom_binary_multichannel(errs, vals, p=p, num_to_ignore=w)
    if gap is not None and gap.sum() > 0:
        for name in list(out.keys()):
            out[f"{name}+gap"] = (out[name] | gap).astype(np.uint8)
        out["gap_only"] = gap
    res = {}
    for name, b in out.items():
        r = official_score(b, ts, f"{raw}/labels.csv", f"{raw}/anomaly_types.csv", chs)
        res[name] = (b, r)
    return res
