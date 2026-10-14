"""Detector A: Telemanom's dynamic threshold + pruning, applied to one channel's errors.

A faithful port of Telemanom-ESA's errors.py (find_epsilon, compare_to_epsilon, prune_anoms),
the mechanism behind ESA's best published Mission-1 result. It is our precision engine:
removing it drops event-wise precision from 0.999 to 0.444.

How it works, per channel:
  1. Threshold search: try eps = mean + z * std for z = 2.5, 3.0, ..., 11.5 and keep the
     best score = (% drop in mean + % drop in std of the errors left below eps)
                  / (number of sequences^2 + number of flagged points),
     allowing at most 5 sequences and < 50 % of points flagged.
  2. Flag points above eps, widen each by +/- error_buffer (100) steps, group into sequences.
  3. Pruning: sort the sequences by their peak error (plus the highest non-flagged error
     as a floor). Walking down that list, any sequence whose peak is less than p (e.g. 30 %)
     above the next one is "not clearly above background" and is removed.
     Example, p = 0.3: peaks [2.0, 1.9, 1.0] + background 0.9 -> drops 5 %, 47 %, 10 %;
     only the trailing run of small drops (1.0 vs 0.9) is removed, 2.0 and 1.9 stay.
"""
import numpy as np
import more_itertools as mit


def _find_epsilon(e_s, mean, sd, sd_lim, error_buffer):
    """Reference threshold search (Telemanom's find_epsilon). Returns the chosen eps."""
    max_score, best_eps = -np.inf, mean + sd_lim * sd
    for z in np.arange(2.5, sd_lim, 0.5):
        eps = mean + sd * z
        pruned = e_s[e_s < eps]                               # errors that stay below
        i = np.argwhere(e_s >= eps).reshape(-1)
        if len(i) == 0:
            break
        buf = np.arange(1, error_buffer)
        i = np.concatenate((i, (i[:, None] + buf).flatten(), (i[:, None] - buf).flatten()))
        i = np.unique(i[(i < len(e_s)) & (i >= 0)])
        E_seq = [(g[0], g[-1]) for g in (list(x) for x in mit.consecutive_groups(i)) if g[0] != g[-1]]
        mpd = (mean - np.mean(pruned)) / mean if mean else 0.0   # % drop in mean
        spd = (sd - np.std(pruned)) / sd if sd else 0.0          # % drop in std
        # big drop is good; many flagged points or many sequences is penalised
        score = (mpd + spd) / (len(E_seq) ** 2 + len(i))
        if score >= max_score and len(E_seq) <= 5 and len(i) < len(e_s) * 0.5:
            max_score, best_eps = score, eps
    return best_eps


def telemanom_detect(e_s, y_vals, p=0.13, sd_lim=12.0, error_buffer=100,
                     min_error_value=0.0, num_to_ignore=0):
    """Detector A for ONE channel (reference version, used by run_m1_lightweight.py).

    Input : e_s (T,) smoothed errors; y_vals (T,) the channel's values (for scale checks);
            p = pruning ratio; num_to_ignore = first rows to skip (RF warm-up)
    Output: (T,) uint8, 1 inside every sequence that survives pruning
    """
    e_s = np.asarray(e_s, float); n = len(e_s)
    mean, sd = e_s.mean(), e_s.std()
    if sd == 0.0:
        return np.zeros(n, np.uint8)
    perc_high, perc_low = np.percentile(y_vals, [95, 5])
    inter_range = perc_high - perc_low
    sd_values = np.std(y_vals)

    eps = _find_epsilon(e_s, mean, sd, sd_lim, error_buffer)

    # guardrail: ignore channels whose errors are tiny relative to the signal's own range
    if not (sd > 0.05 * sd_values or e_s.max() > 0.05 * inter_range) \
            or n < error_buffer or not e_s.max() > min_error_value:
        return np.zeros(n, np.uint8)

    i = np.argwhere((e_s >= eps) & (e_s > 0.05 * inter_range)).reshape(-1)
    if len(i) == 0:
        return np.zeros(n, np.uint8)
    buf = np.arange(1, error_buffer + 1)
    i = np.concatenate((i, (i[:, None] + buf).flatten(), (i[:, None] - buf).flatten()))
    i = i[(i < n) & (i >= 0)]
    i = np.unique(i[i >= num_to_ignore])
    if len(i) == 0:
        return np.zeros(n, np.uint8)

    cand = np.setdiff1d(np.arange(n), i)
    non_anom_max = float(np.max(e_s[cand])) if len(cand) else 0.0   # background level
    E_seq = [(g[0], g[-1]) for g in (list(x) for x in mit.consecutive_groups(i)) if g[0] != g[-1]]
    if not E_seq:
        return np.zeros(n, np.uint8)

    # pruning: keep only sequences clearly above the next one / the background
    E_seq_max = np.array([e_s[a:b + 1].max() for a, b in E_seq])
    order = np.append(np.sort(E_seq_max)[::-1], [non_anom_max])
    to_remove = []
    for k in range(len(order) - 1):
        if (order[k] - order[k + 1]) / order[k] < p:
            to_remove.append(int(np.argwhere(E_seq_max == order[k])[0, 0]))
        else:
            to_remove = []              # a big drop: everything above it is kept
    keep = [E_seq[k] for k in range(len(E_seq)) if k not in to_remove]

    out = np.zeros(n, np.uint8)
    for a, b in keep:
        out[a:b + 1] = 1
    return out


def _dilate(hit, k):
    """Mask of every index within k steps of a True in `hit`, computed with a cumulative sum.
    Same result as listing hit-indices +/- 1..k, but memory O(T) however many hits there are."""
    n = len(hit)
    c = np.concatenate(([0], np.cumsum(hit, dtype=np.int64)))
    j = np.arange(n)
    return (c[np.minimum(j + k + 1, n)] - c[np.maximum(j - k, 0)]) > 0


def _runs(mask):
    """(starts, ends) of every run of True, both inclusive."""
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def _find_epsilon_fast(e_s, mean, sd, sd_lim, error_buffer):
    """_find_epsilon with identical output and O(T) memory (uses _dilate / _runs)."""
    max_score, best_eps = -np.inf, mean + sd_lim * sd
    for z in np.arange(2.5, sd_lim, 0.5):
        eps = mean + sd * z
        pruned = e_s[e_s < eps]
        hit = e_s >= eps
        if not hit.any():
            break
        m = _dilate(hit, error_buffer - 1)
        n_i = int(m.sum())                          # = len(i) in the reference
        a, b = _runs(m)
        n_seq = int(np.count_nonzero(a != b))       # = len(E_seq) in the reference
        mpd = (mean - np.mean(pruned)) / mean if mean else 0.0
        spd = (sd - np.std(pruned)) / sd if sd else 0.0
        score = (mpd + spd) / (n_seq ** 2 + n_i)
        if score >= max_score and n_seq <= 5 and n_i < len(e_s) * 0.5:
            max_score, best_eps = score, eps
    return best_eps


def telemanom_detect_multi(e_s, y_vals, ps, sd_lim=12.0, error_buffer=100,
                           min_error_value=0.0, num_to_ignore=0):
    """telemanom_detect for several pruning ratios at once (used by run_full_channels.py).

    Input : as telemanom_detect, but ps = list of p values
    Output: {p: (T,) uint8}, each identical to telemanom_detect(..., p=p)
    The threshold search (most of the cost) runs once for all p, and the +/- buffer is a
    dilated mask instead of a huge index array (that array OOM-killed full-scale runs).
    """
    e_s = np.asarray(e_s, float); n = len(e_s)
    empty = lambda: {p: np.zeros(n, np.uint8) for p in ps}
    mean, sd = e_s.mean(), e_s.std()
    if sd == 0.0:
        return empty()
    perc_high, perc_low = np.percentile(y_vals, [95, 5])
    inter_range = perc_high - perc_low
    sd_values = np.std(y_vals)

    eps = _find_epsilon_fast(e_s, mean, sd, sd_lim, error_buffer)

    if not (sd > 0.05 * sd_values or e_s.max() > 0.05 * inter_range) \
            or n < error_buffer or not e_s.max() > min_error_value:
        return empty()

    hit = (e_s >= eps) & (e_s > 0.05 * inter_range)
    if not hit.any():
        return empty()
    m = _dilate(hit, error_buffer)
    m[:num_to_ignore] = False
    if not m.any():
        return empty()

    non_anom_max = float(np.max(e_s[~m])) if (~m).any() else 0.0
    a, b = _runs(m)
    keep_seq = a != b                              # single points are not sequences
    a, b = a[keep_seq], b[keep_seq]
    if len(a) == 0:
        return empty()

    E_seq_max = np.array([e_s[s:t + 1].max() for s, t in zip(a, b)])
    order = np.append(np.sort(E_seq_max)[::-1], [non_anom_max])
    # Same rule as the reference loop: what gets removed is the trailing run of small drops;
    # each removed peak maps to the FIRST sequence with that peak value.
    with np.errstate(divide="ignore", invalid="ignore"):
        drop = (order[:-1] - order[1:]) / order[:-1]
    uniq, first = np.unique(E_seq_max, return_index=True)
    first_of = dict(zip(uniq.tolist(), first.tolist()))
    res = {}
    for p in ps:
        small = drop < p
        fails = np.flatnonzero(~small)
        start = fails[-1] + 1 if len(fails) else 0
        to_remove = {first_of[v] for v in order[start:len(order) - 1].tolist()}
        out = np.zeros(n, np.uint8)
        for k in range(len(a)):
            if k not in to_remove:
                out[a[k]:b[k] + 1] = 1
        res[p] = out
    return res
