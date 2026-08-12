"""GDNv3 detection stack — per-channel, locally normalized, multi-span, merged.

Building blocks (each addresses a measured GDNv2 failure):
  block_z        : LOCAL (block-wise, interpolated) robust normalization.
                   GDNv2 used GLOBAL median/IQR over 7 years; every missed event sat
                   below the global p99.9 noise floor. Local baselines lift events that
                   occur in quiet stretches.
  multi_span_z   : max over EWMA spans {1,3,15,60,240}. GDNv2 used one span (5) and
                   caught 0/12 events of length 6-60.
  close_runs     : morphological closing -> merges fragmented alarms.
                   GDNv2 alarming_precision was 0.228 (~4.4 predicted events per real one).
  gap_pred_ch    : per-channel gap detector (count==0 runs >= L).
"""
import numpy as np, pandas as pd


def ewma(errs, span):
    if span <= 1:
        return errs.astype(np.float32)
    return pd.DataFrame(errs).ewm(span=span, adjust=False).mean().to_numpy(np.float32)


def block_stats(e, block):
    """Per-block median and IQR, linearly interpolated back to full length. e: (T,) float."""
    T = len(e)
    nb = max(int(np.ceil(T / block)), 2)
    idx = np.minimum(np.arange(nb) * block + block // 2, T - 1)
    med = np.empty(nb, np.float32)
    iqr = np.empty(nb, np.float32)
    for i in range(nb):
        seg = e[i * block: min((i + 1) * block, T)]
        if len(seg) == 0:
            med[i], iqr[i] = 0.0, 1.0
            continue
        q1, q2, q3 = np.percentile(seg, [25, 50, 75])
        med[i], iqr[i] = q2, max(q3 - q1, 1e-6)
    x = np.arange(T)
    return np.interp(x, idx, med).astype(np.float32), np.interp(x, idx, iqr).astype(np.float32)


def block_z(errs, block, floor_q=0.25):
    """Locally normalized robust z, per channel. Returns (T, C).
    IQR is floored at a global quantile so ultra-quiet blocks cannot manufacture
    enormous z-scores from numerical noise."""
    T, C = errs.shape
    Z = np.empty((T, C), np.float32)
    for c in range(C):
        med, iqr = block_stats(errs[:, c], block)
        gfloor = max(float(np.quantile(iqr, floor_q)), 1e-6)
        Z[:, c] = (errs[:, c] - med) / np.maximum(iqr, gfloor)
    return Z


def multi_span_z(errs, spans, block, floor_q=0.25):
    """max over spans of locally normalized z. Returns (T, C)."""
    out = None
    for sp in spans:
        z = block_z(ewma(errs, sp), block, floor_q)
        out = z if out is None else np.maximum(out, z)
    return out


def runs_of(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.stack([np.where(d == 1)[0], np.where(d == -1)[0]], 1)


def close_runs(pred, max_gap):
    """Morphological closing: bridge 0-gaps shorter than max_gap between positives."""
    if max_gap <= 0 or pred.sum() == 0:
        return pred
    out = pred.copy()
    r = runs_of(pred > 0)
    for i in range(len(r) - 1):
        if r[i + 1, 0] - r[i, 1] <= max_gap:
            out[r[i, 1]:r[i + 1, 0]] = 1
    return out


def open_runs(pred, min_len):
    """Drop positive runs shorter than min_len (removes isolated single-sample blips)."""
    if min_len <= 1 or pred.sum() == 0:
        return pred
    out = np.zeros_like(pred)
    for s, e in runs_of(pred > 0):
        if e - s >= min_len:
            out[s:e] = 1
    return out


def gap_pred_ch(count, L):
    """Per-channel gap detector: count==0 runs of length >= L. count: (T, C)."""
    T, C = count.shape
    out = np.zeros((T, C), np.uint8)
    for c in range(C):
        for s, e in runs_of(count[:, c] == 0):
            if e - s >= L:
                out[s:e, c] = 1
    return out


def zero_gap_margin(errs, gap_any, margin):
    """Zero errors inside gap runs + margin (re-acquisition spikes poison thresholds)."""
    out = errs.copy()
    for s, e in runs_of(gap_any):
        out[s:min(e + margin, len(out))] = 0.0
    return out


def build_pred(Z, thr, close_gap, min_len, gapdet=None, tc_boost=None, thr_boost=None):
    """Per-channel prediction from z-scores + optional telecommand-gated lower threshold."""
    P = (Z > thr).astype(np.uint8)
    if tc_boost is not None and thr_boost is not None:
        P = np.maximum(P, ((Z > thr_boost) & tc_boost[:, None]).astype(np.uint8))
    if gapdet is not None:
        P = np.maximum(P, gapdet)
    for c in range(P.shape[1]):
        col = close_runs(P[:, c], close_gap)
        P[:, c] = open_runs(col, min_len)
    return P
