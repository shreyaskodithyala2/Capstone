"""Faithful port of Telemanom-ESA dynamic thresholding + pruning (errors.py find_epsilon
+ compare_to_epsilon + prune_anoms). This is the mechanism behind Telemanom-Pruned's 0.786
(precision 0.999). Applied per-channel to GDN's smoothed forecast errors."""
import numpy as np
import more_itertools as mit


def _find_epsilon(e_s, mean, sd, sd_lim, error_buffer):
    max_score, best_eps = -np.inf, mean + sd_lim * sd
    for z in np.arange(2.5, sd_lim, 0.5):
        eps = mean + sd * z
        pruned = e_s[e_s < eps]
        i = np.argwhere(e_s >= eps).reshape(-1)
        if len(i) == 0:
            break
        buf = np.arange(1, error_buffer)
        i = np.concatenate((i, (i[:, None] + buf).flatten(), (i[:, None] - buf).flatten()))
        i = np.unique(i[(i < len(e_s)) & (i >= 0)])
        E_seq = [(g[0], g[-1]) for g in (list(x) for x in mit.consecutive_groups(i)) if g[0] != g[-1]]
        mpd = (mean - np.mean(pruned)) / mean if mean else 0.0
        spd = (sd - np.std(pruned)) / sd if sd else 0.0
        score = (mpd + spd) / (len(E_seq) ** 2 + len(i))
        if score >= max_score and len(E_seq) <= 5 and len(i) < len(e_s) * 0.5:
            max_score, best_eps = score, eps
    return best_eps


def telemanom_detect(e_s, y_vals, p=0.13, sd_lim=12.0, error_buffer=100,
                     min_error_value=0.0, num_to_ignore=0):
    """Return per-timestep binary anomalies for ONE channel's error stream e_s."""
    e_s = np.asarray(e_s, float); n = len(e_s)
    mean, sd = e_s.mean(), e_s.std()
    if sd == 0.0:
        return np.zeros(n, np.uint8)
    perc_high, perc_low = np.percentile(y_vals, [95, 5])
    inter_range = perc_high - perc_low
    sd_values = np.std(y_vals)

    eps = _find_epsilon(e_s, mean, sd, sd_lim, error_buffer)

    # scale guardrail (errors.py compare_to_epsilon)
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
    non_anom_max = float(np.max(e_s[cand])) if len(cand) else 0.0
    E_seq = [(g[0], g[-1]) for g in (list(x) for x in mit.consecutive_groups(i)) if g[0] != g[-1]]
    if not E_seq:
        return np.zeros(n, np.uint8)

    # prune: drop the leading run of insufficiently-separated sequences
    E_seq_max = np.array([e_s[a:b + 1].max() for a, b in E_seq])
    order = np.append(np.sort(E_seq_max)[::-1], [non_anom_max])
    to_remove = []
    for k in range(len(order) - 1):
        if (order[k] - order[k + 1]) / order[k] < p:
            to_remove.append(int(np.argwhere(E_seq_max == order[k])[0, 0]))
        else:
            to_remove = []
    keep = [E_seq[k] for k in range(len(E_seq)) if k not in to_remove]

    out = np.zeros(n, np.uint8)
    for a, b in keep:
        out[a:b + 1] = 1
    return out


def telemanom_binary_multichannel(errs, y_test, p=0.13, sd_lim=12.0, error_buffer=100,
                                  min_error_value=0.0, num_to_ignore=0):
    """errs, y_test: (T, N). Return global binary series = OR of per-channel detections."""
    T, N = errs.shape
    out = np.zeros(T, np.uint8)
    for c in range(N):
        out |= telemanom_detect(errs[:, c], y_test[:, c], p, sd_lim, error_buffer,
                                min_error_value, num_to_ignore)
    return out
