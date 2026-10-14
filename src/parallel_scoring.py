"""Run ESA's metric on several CPU cores during the full-set TRAIN search.

ESA's ESAScores is plain CPU code (pandas + interval arithmetic) and is the bottleneck of
the search, so predictions are scored on a pool of worker processes. The metric itself is
called unchanged, through esa_metrics.global_f05.

Workers are started with "spawn" (fresh interpreters: no inherited CUDA state, no copies of
the parent's multi-GB arrays). Timestamps and labels are sent to each worker once.
"""
import multiprocessing as mp
import pandas as pd

_W = {}                   # per-worker state, filled by _init


def _init(ts, raw, channels, labels_df, preds=None):
    """Worker start-up: import the metric code once and store the shared inputs."""
    from esa_metrics import ESAScores, ChannelAwareFScore, ADTQC   # noqa: F401 (warm import)
    _W.update(ts=pd.to_datetime(pd.Series(ts)).reset_index(drop=True), raw=raw,
              channels=channels, labels_df=labels_df, preds=preds)


def make_pool(workers, ts, raw, channels, labels_df, preds=None):
    """Start `workers` spawn processes, each initialised with timestamps + labels.
    Use with pool.apply_async(score1d, (pred,))."""
    return mp.get_context("spawn").Pool(workers, initializer=_init,
                                        initargs=(ts, raw, channels, labels_df, preds))


def score1d(pred):
    """Worker task: global event-wise F0.5 of a (T,) uint8 prediction (== esa_metrics.global_f05)."""
    from esa_metrics import global_f05
    r = global_f05(pred[:, None], _W["ts"], _W["raw"], _W["channels"],
                   labels_df=_W["labels_df"], compress=True)
    return r["global_EW_F_0.50"]
