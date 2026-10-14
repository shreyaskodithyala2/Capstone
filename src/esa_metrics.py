"""Scores binary predictions with ESA-ADB's official metric classes (all five aspects).

We call ESA's own metric code (timeeval.metrics, from the ESA-ADB repo) and replicate how
its benchmark harness prepares the inputs (timeeval/core/experiments.py), so our numbers are
directly comparable to published ones. Validated: re-scoring ESA's official Docker output
reproduces its 0.9487 / 0.9509 / 0.94 exactly (tests/test_scorer_matches_official.py).

Inputs are always: per-channel 0/1 predictions (T, C) + the T grid timestamps.
  Aspect 1  event-wise F0.5        global prediction = OR over channels      ESAScores
  Aspect 2  channel/subsystem F0.5 did the right channels fire, per event   ChannelAwareFScore
  Aspect 3  alarming precision     one alarm per event, not many             ESAScores
  Aspect 4  ADTQC                  alarm timing relative to event onset      ADTQC
  Aspect 5  affiliation F0.5       how close alarms are to events in time    ESAScores
"""
import contextlib, io, sys
import numpy as np, pandas as pd

from config import TIMEEVAL_PATH

sys.path.insert(0, TIMEEVAL_PATH)          # ESA-ADB clone (or the vendored copy on Kaggle)
from timeeval.metrics import ESAScores, ChannelAwareFScore, ADTQC

BETA = 0.5                                       # F0.5: precision weighted 4x over recall
SEL = {"Category": ["Rare Event", "Anomaly"]}    # headline: score anomalies + rare events
SEL_ANOM = {"Category": ["Anomaly"]}             # stricter view ESA also reports


def change_points(pred):
    """Rows where a 0/1 series changes value, plus the first and last row.

    ESA's metrics only look at where alarms start and stop, so scoring these rows gives the
    same result as scoring all T rows, at a fraction of the memory (~1 GB -> ~MB per channel).
    Example: [0, 0, 1, 1, 1, 0, 0] -> rows [0, 2, 5, 6]
    """
    pred = np.asarray(pred)
    keep = np.empty(len(pred), bool)
    keep[0] = True
    np.not_equal(pred[1:], pred[:-1], out=keep[1:])
    keep[-1] = True
    return np.flatnonzero(keep)


def _series(ts, pred, compress):
    """Format one 0/1 series the way ESA's metrics expect: DataFrame[Timestamp, Score]."""
    pred = np.asarray(pred).astype(np.uint8)
    if not compress:
        return pd.DataFrame({"Timestamp": ts, "Score": pred})
    i = change_points(pred)
    return pd.DataFrame({"Timestamp": ts.iloc[i].reset_index(drop=True), "Score": pred[i]})


def build_labels_df(labels_csv, anomaly_types_csv, ts):
    """Ground-truth table, built exactly like ESA's harness does.

    Input : labels.csv (ID, Channel, StartTime, EndTime), anomaly_types.csv, scored timestamps
    Output: one row per (event, channel) for events FULLY inside the scored time range, with
            Category / Dimensionality / Locality / Length joined in by event ID
    """
    lab = pd.read_csv(labels_csv, parse_dates=["StartTime", "EndTime"])
    lab["StartTime"] = lab["StartTime"].apply(lambda t: t.tz_localize(None) if t.tzinfo else t)
    lab["EndTime"] = lab["EndTime"].apply(lambda t: t.tz_localize(None) if t.tzinfo else t)
    lab = lab[lab["StartTime"] >= ts.min()]
    lab = lab[lab["EndTime"] <= ts.max()]
    at = pd.read_csv(anomaly_types_csv)
    cols = list(at.columns[-4:])                      # Category, Dimensionality, Locality, Length
    for c in cols:
        lab[c] = ""
    for _, row in at.iterrows():
        lab.loc[lab["ID"] == row["ID"], cols] = row[cols].values
    return lab


def subsystems_mapping(channels_csv, channels):
    """{subsystem name: [channels]} from channels.csv, for the subsystem-aware F-score."""
    ch = pd.read_csv(channels_csv)
    ch = ch[ch["Channel"].isin(channels)]
    return {s: [*v] for s, v in ch.groupby("Subsystem")["Channel"]}


def global_f05(pred_per_channel, ts, raw_dir, channels, labels_df=None, compress=False):
    """Fast path for tuning: only the global ESAScores (event-wise F0.5 and friends).

    Input : pred_per_channel (T, C) 0/1; ts (T,) timestamps; raw_dir = mission folder with
            labels.csv / anomaly_types.csv; channels = names of the C columns
    Output: {"global_EW_F_0.50": ..., "global_EW_precision": ..., "global_EW_recall": ...,
             "global_alarming_precision": ..., "global_AFF_F_0.50": ..., ...}
    """
    labels_csv = f"{raw_dir}/labels.csv"; at_csv = f"{raw_dir}/anomaly_types.csv"
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    if labels_df is None:
        labels_df = build_labels_df(labels_csv, at_csv, ts)
    y_true = labels_df[labels_df["Channel"].isin(channels)]        # NOT de-duplicated (as ESA)
    scores = _series(ts, pred_per_channel.max(axis=1), compress)   # OR over channels
    # ESA API: ESAScores(...).score(labels without the Channel column, DataFrame[Timestamp,
    # Score]) -> dict of event-wise / alarming / affiliation metrics. It prints debug text,
    # which is silenced here.
    with contextlib.redirect_stdout(io.StringIO()):
        r = ESAScores(betas=BETA, select_labels=SEL).score(y_true.drop(columns=["Channel"]), scores)
    return {f"global_{k}": float(v) for k, v in r.items()}


def full_scoreboard(pred_per_channel, ts, raw_dir, channels, quiet=True, compress=False):
    """Every official metric for one set of test predictions.

    Input : pred_per_channel (T, C) uint8, column order == `channels`; ts; raw_dir
    Output: flat dict. Keys: global_* (aspects 1, 3, 5), chan_* (aspect 2), adtqc_*
            (aspect 4), perchannel_EW_F_0.50_mean, the same under anomaly-only labels
            (anom_*), and n_positive_rows.
    """
    labels_csv = f"{raw_dir}/labels.csv"
    at_csv = f"{raw_dir}/anomaly_types.csv"
    ch_csv = f"{raw_dir}/channels.csv"
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)

    labels_df = build_labels_df(labels_csv, at_csv, ts)
    y_true = labels_df[labels_df["Channel"].isin(channels)]

    y_pred = {c: _series(ts, pred_per_channel[:, i], compress) for i, c in enumerate(channels)}
    glob_scores = _series(ts, pred_per_channel.max(axis=1), compress)

    out, ctx = {}, (contextlib.redirect_stdout(io.StringIO()) if quiet
                    else contextlib.nullcontext())
    with ctx:
        # aspects 1, 3, 5: one global 0/1 series vs all events
        m = ESAScores(betas=BETA, select_labels=SEL)
        for k, v in m.score(y_true.drop(columns=["Channel"]), glob_scores).items():
            out[f"global_{k}"] = float(v)

        # aspect 2: ChannelAwareFScore.score(labels WITH Channel, {channel: series},
        # {subsystem: [channels]}) -> channel- and subsystem-aware F0.5
        sub_map = subsystems_mapping(ch_csv, channels)
        ca = ChannelAwareFScore(beta=BETA, select_labels=SEL)
        for k, v in ca.score(y_true.copy(), {c: d.copy() for c, d in y_pred.items()},
                             sub_map).items():
            out[f"chan_{k}"] = float(v)

        # aspect 4: ADTQC.score(labels, {channel: series}) -> timing-quality scores
        try:
            ad = ADTQC(select_labels=SEL)
            for k, v in ad.score(y_true.copy(), {c: d.copy() for c, d in y_pred.items()}).items():
                out[f"adtqc_{k}"] = float(v)
        except Exception as e:
            out["adtqc_error"] = str(e)[:120]

        # diagnostic only: event-wise F0.5 of each channel on its own events, averaged
        per = []
        for i, c in enumerate(channels):
            yt = y_true[y_true["Channel"] == c].drop(columns=["Channel"])
            if len(yt) == 0:                    # channel has no events in the test period
                continue
            sc = _series(ts, pred_per_channel[:, i], compress)
            try:
                per.append(ESAScores(betas=BETA, select_labels=SEL).score(yt, sc)[f"EW_F_{BETA:.2f}"])
            except Exception:
                pass
        if per:
            out["perchannel_EW_F_0.50_mean"] = float(np.mean(per))

        # the same three metrics with anomaly-only labels (separate anom_* keys)
        try:
            for k, v in ESAScores(betas=BETA, select_labels=SEL_ANOM).score(
                    y_true.drop(columns=["Channel"]), glob_scores).items():
                out[f"anom_global_{k}"] = float(v)
            ca2 = ChannelAwareFScore(beta=BETA, select_labels=SEL_ANOM)
            for k, v in ca2.score(y_true.copy(), {c: d.copy() for c, d in y_pred.items()},
                                  sub_map).items():
                out[f"anom_chan_{k}"] = float(v)
            for k, v in ADTQC(select_labels=SEL_ANOM).score(
                    y_true.copy(), {c: d.copy() for c, d in y_pred.items()}).items():
                out[f"anom_adtqc_{k}"] = float(v)
        except Exception as e:
            out["anom_error"] = str(e)[:160]
    out["n_positive_rows"] = int(pred_per_channel.max(axis=1).sum())
    return out


# Metrics printed first in run logs.
PRIMARY = ["global_EW_F_0.50", "global_EW_precision", "global_EW_recall",
           "global_alarming_precision", "global_AFF_F_0.50",
           "chan_channel_F0.50", "chan_subsystem_F0.50",
           "adtqc_Total", "adtqc_AfterRate", "perchannel_EW_F_0.50_mean"]
