"""Detection building blocks: turn forecast errors into 0/1 alarms, per channel.

Where each piece sits in the pipeline (see run_m1_lightweight.py / run_full_channels.py):
  step 1      zero_gap_margin   zero the errors inside telemetry outages (+30 steps)
  smoothing   ewma              exponentially weighted moving average of the errors
  Detector A  -> telemanom_detector.py (dynamic threshold + pruning)
  Detector B  multi_span_z      robust z-score of smoothed errors, max over 5 smoothing spans
  Detector C  gap_pred_*        flag long runs of "no telemetry arrived" (count == 0)
  Detector D  -> assemble() in the runners (lower Detector-B threshold near a telecommand)
  cleanup     close_runs        merge alarms separated by short gaps
              open_runs         drop alarms shorter than min_len

Shapes: errors / z / predictions are (T, C) = (timesteps, channels); a "run" is a maximal
stretch of consecutive 1s, stored as [start, end) row indices.
"""
import numpy as np, pandas as pd


def ewma(errs, span):
    """Exponentially weighted moving average along time, per channel.

    Input : errs (T,) or (T, C); span = smoothing length (alpha = 2 / (span + 1))
    Output: same shape, float32. span <= 1 returns the input unchanged.
    Example: span 5 turns a single-step spike of 1.0 into 0.33, 0.22, 0.15, ... (decaying)
    """
    if span <= 1:
        return errs.astype(np.float32)
    return pd.DataFrame(errs).ewm(span=span, adjust=False).mean().to_numpy(np.float32)


def block_stats(e, block):
    """Local median and IQR: computed per block of `block` rows, then linearly interpolated
    between block centres so the baseline changes smoothly.

    Input : e (T,) or (T, C); block = rows per block (2880 = 1 day on M1's 30 s grid)
    Output: (median, iqr), each the same shape as e
    Note  : forces at least 2 blocks; with block >= T this adds an empty phantom block
            (median 0, IQR 1) -- use global_multi_span_z for a true whole-series baseline.
    """
    a = e[:, None] if e.ndim == 1 else e
    T, C = a.shape
    nb = max(int(np.ceil(T / block)), 2)
    idx = np.minimum(np.arange(nb) * block + block // 2, T - 1)       # block centres
    med = np.empty((nb, C), np.float32)
    iqr = np.empty((nb, C), np.float32)
    for i in range(nb):
        seg = a[i * block: min((i + 1) * block, T)]
        if len(seg) == 0:
            med[i], iqr[i] = 0.0, 1.0
            continue
        q1, q2, q3 = np.percentile(seg, [25, 50, 75], axis=0)        # all channels at once
        med[i], iqr[i] = q2, np.maximum(q3 - q1, 1e-6)
    x = np.arange(T)
    M = np.empty((T, C), np.float32)
    I = np.empty((T, C), np.float32)
    for c in range(C):                       # np.interp is 1-D only
        M[:, c] = np.interp(x, idx, med[:, c])
        I[:, c] = np.interp(x, idx, iqr[:, c])
    return (M[:, 0], I[:, 0]) if e.ndim == 1 else (M, I)


def block_z(errs, block, floor_q=0.25):
    """Robust z-score against the LOCAL baseline: z = (error - local median) / local IQR.

    Input : errs (T, C); block as in block_stats; floor_q = IQR floor quantile
    Output: z (T, C) float32
    The IQR is floored at each channel's own 25th-percentile IQR, so an unusually quiet
    block cannot turn tiny numerical noise into a huge z.
    """
    med, iqr = block_stats(errs, block)
    # kept float32 on purpose: float64 here shifts large z by ~5e-4, enough to move points
    # across the 0.9999-quantile threshold and change the published scores
    gfloor = np.maximum(np.quantile(iqr, floor_q, axis=0), 1e-6).astype(np.float32)
    return ((errs - med) / np.maximum(iqr, gfloor)).astype(np.float32)


def multi_span_z(errs, spans, block, floor_q=0.25, chunk=16):
    """Detector B input: for each smoothing span, smooth -> local robust z; keep the max.

    Input : errs (T, C) gap-cleaned errors; spans e.g. [1, 3, 15, 60, 240]
    Output: Z (T, C) float32
    Why several spans: span 1 catches 1-step spikes, span 240 catches slow shifts; one span
    (5) alone missed all 12 events of length 6-60 steps.
    Channels are processed `chunk` at a time to bound memory (~9 GB otherwise at full scale).
    """
    T, C = errs.shape
    out = np.empty((T, C), np.float32)
    for c0 in range(0, C, chunk):
        sub = errs[:, c0:min(c0 + chunk, C)]
        acc = None
        for sp in spans:
            z = block_z(ewma(sub, sp), block, floor_q)
            acc = z if acc is None else np.maximum(acc, z, out=acc)
        out[:, c0:c0 + acc.shape[1]] = acc
    return out


def global_multi_span_z(errs, spans, chunk=16):
    """Same as multi_span_z but with ONE median/IQR per channel over the whole series.

    Use this rather than multi_span_z(errs, spans, block=len(errs)), which hits the phantom
    block in block_stats and deflates z over the second half of the series (up to ~60x).
    """
    T, C = errs.shape
    out = np.empty((T, C), np.float32)
    for c0 in range(0, C, chunk):
        sub = errs[:, c0:min(c0 + chunk, C)]
        acc = None
        for sp in spans:
            e = ewma(sub, sp)
            q1, q2, q3 = np.percentile(e, [25, 50, 75], axis=0)
            iqr = np.maximum(q3 - q1, 1e-6)
            z = ((e - q2.astype(np.float32)) / iqr.astype(np.float32)).astype(np.float32)
            acc = z if acc is None else np.maximum(acc, z, out=acc)
        out[:, c0:c0 + acc.shape[1]] = acc
    return out


def runs_of(mask):
    """Start/end of every run of True in a 1-D mask.

    Output: (R, 2) int array of [start, end) pairs
    Example: [0, 1, 1, 0, 1] -> [[1, 3], [4, 5]]
    """
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.stack([np.where(d == 1)[0], np.where(d == -1)[0]], 1)


def close_runs(pred, max_gap):
    """Merge alarms whose gap of 0s is <= max_gap (morphological closing), one channel.

    Example: max_gap=2, [1,1,0,0,1,0,0,0,1] -> [1,1,1,1,1,0,0,0,1]
    Effect : one real event no longer counts as several predicted events.
    """
    if max_gap <= 0 or pred.sum() == 0:
        return pred
    out = pred.copy()
    r = runs_of(pred > 0)
    for i in range(len(r) - 1):
        if r[i + 1, 0] - r[i, 1] <= max_gap:            # gap between run i and run i+1
            out[r[i, 1]:r[i + 1, 0]] = 1
    return out


def open_runs(pred, min_len):
    """Drop alarms shorter than min_len steps, one channel.

    Example: min_len=3, [1,0,1,1,1,0,1,1] -> [0,0,1,1,1,0,0,0]
    """
    if min_len <= 1 or pred.sum() == 0:
        return pred
    out = np.zeros_like(pred)
    for s, e in runs_of(pred > 0):
        if e - s >= min_len:
            out[s:e] = 1
    return out


def gap_pred_ch(count, L):
    """Detector C: flag every run of count == 0 (no telemetry) lasting >= L steps.

    Input : count (T, C) raw samples per grid slot; L = minimum silent run (10 on M1)
    Output: (T, C) uint8, 1 over the whole silent run
    Example: L=3, count [1,0,0,1,0,0,0,1] -> [0,0,0,0,1,1,1,0]
    """
    T, C = count.shape
    out = np.zeros((T, C), np.uint8)
    for c in range(C):
        for s, e in runs_of(count[:, c] == 0):
            if e - s >= L:
                out[s:e, c] = 1
    return out


def zero_gap_margin(errs, gap_any, margin):
    """Step 1 (before every detector): set errors to 0 where ALL channels are silent, plus
    `margin` steps after. When telemetry returns, the first forecasts are wildly wrong
    (z up to ~1149); those spikes would otherwise dominate Detector A's threshold.

    Input : errs (T, C); gap_any (T,) bool = all channels have count == 0; margin = 30
    Output: copy of errs with those rows zeroed
    """
    out = errs.copy()
    for s, e in runs_of(gap_any):
        out[s:min(e + margin, len(out))] = 0.0
    return out


# ---- all-channel versions of the cleanup steps, used by run_full_channels.py ----
# A run-table loop per channel; benchmarked 3x faster than scipy.ndimage morphology here.

def close_runs2d(pred, max_gap):
    """close_runs on every column. pred (T, C) -> (T, C) uint8."""
    if max_gap <= 0:
        return pred.astype(np.uint8)
    out = np.empty(pred.shape, np.uint8)
    for c in range(pred.shape[1]):
        out[:, c] = close_runs(pred[:, c].astype(np.uint8), max_gap)
    return out


def open_runs2d(pred, min_len):
    """open_runs on every column. pred (T, C) -> (T, C) uint8."""
    if min_len <= 1:
        return pred.astype(np.uint8)
    out = np.empty(pred.shape, np.uint8)
    for c in range(pred.shape[1]):
        out[:, c] = open_runs(pred[:, c].astype(np.uint8), min_len)
    return out


def gap_pred_ch2d(count, L):
    """gap_pred_ch written as open_runs2d on the silence mask (same output)."""
    return open_runs2d((count == 0).astype(np.uint8), L)


def channel_gap_cadence(count):
    """Typical silence length of each channel = median length of its count == 0 runs.

    Input : count (T, C) from the TRAIN split (label-free: it describes the sampling schedule)
    Output: (C,) float32
    Example: a channel sampled every ~15 min on a 30 s grid has cadence ~29.
    """
    C = count.shape[1]
    out = np.ones(C, np.float32)
    for c in range(C):
        r = runs_of(count[:, c] == 0)
        if len(r):
            out[c] = float(np.median(r[:, 1] - r[:, 0]))
    return out


def gap_pred_adaptive(count, L_global, cadence=None, mult=None):
    """Detector C for the full channel sets: a per-channel minimum silence length.

        L_c = max(L_global, ceil(mult * cadence_c))

    A flat L=10 breaks at full scale: channel_61 is silent 96.7 % of the time simply because
    it is sampled every ~15 min, so it would alarm almost always. Scaling L by each channel's
    own cadence (fitted on train) fixes that. mult=None gives the flat-L detector.
    Example: L_global=10, mult=6 -> channel_41 (cadence 5) needs 30 silent steps,
             channel_61 (cadence 29) needs 174.
    """
    if cadence is None or mult is None:
        return gap_pred_ch2d(count, L_global)
    Lc = np.maximum(L_global, np.ceil(mult * cadence)).astype(int)
    out = np.zeros(count.shape, np.uint8)
    for c in range(count.shape[1]):
        for s, e in runs_of(count[:, c] == 0):
            if e - s >= Lc[c]:
                out[s:e, c] = 1
    return out
