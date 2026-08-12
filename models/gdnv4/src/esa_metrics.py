"""Complete ESA-ADB scoreboard — ALL FIVE evaluation aspects.

Faithful to timeeval/core/experiments.py:
  * labels_df  : read labels.csv, tz-strip, keep events FULLY INSIDE the scored range,
                 merge the last four anomaly_types columns by ID (lines 339-355)
  * per-channel filter WITHOUT deduplication                        (line 119)
  * global ESAScores : y_true.drop(columns=["Channel"]) + max-over-channels scores (line 155)
  * ChannelAwareFScore(y_true, y_pred_dict, subsystems_mapping)      (line 206)
  * ADTQC(y_true, y_pred_dict)                                       (line 208)
  * per-channel ESAScores                                            (line 227)

y_pred_dict values are DataFrames with columns [Timestamp, Score] (uint8), exactly as
the harness constructs them.
"""
import contextlib, io, sys
import numpy as np, pandas as pd

sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/ESA-ADB")
from timeeval.metrics import ESAScores, ChannelAwareFScore, ADTQC

BETA = 0.5
SEL = {"Category": ["Rare Event", "Anomaly"]}


def build_labels_df(labels_csv, anomaly_types_csv, ts):
    """Replicates Experiment.labels_df construction."""
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
    ch = pd.read_csv(channels_csv)
    ch = ch[ch["Channel"].isin(channels)]
    return {s: [*v] for s, v in ch.groupby("Subsystem")["Channel"]}


def global_f05(pred_per_channel, ts, raw_dir, channels, labels_df=None):
    """FAST path for hyper-parameter search: global ESAScores only (1 metric, not 14).
    Identical to the `global_*` block of full_scoreboard."""
    labels_csv = f"{raw_dir}/labels.csv"; at_csv = f"{raw_dir}/anomaly_types.csv"
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    if labels_df is None:
        labels_df = build_labels_df(labels_csv, at_csv, ts)
    y_true = labels_df[labels_df["Channel"].isin(channels)]
    scores = pd.DataFrame({"Timestamp": ts, "Score": pred_per_channel.max(axis=1).astype(np.uint8)})
    with contextlib.redirect_stdout(io.StringIO()):
        r = ESAScores(betas=BETA, select_labels=SEL).score(y_true.drop(columns=["Channel"]), scores)
    return {f"global_{k}": float(v) for k, v in r.items()}


def full_scoreboard(pred_per_channel, ts, raw_dir, channels, quiet=True):
    """pred_per_channel: (T, n_ch) uint8 binary predictions, column order == `channels`.
    Returns a flat dict of every official metric."""
    labels_csv = f"{raw_dir}/labels.csv"
    at_csv = f"{raw_dir}/anomaly_types.csv"
    ch_csv = f"{raw_dir}/channels.csv"
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)

    labels_df = build_labels_df(labels_csv, at_csv, ts)
    y_true = labels_df[labels_df["Channel"].isin(channels)]

    y_pred = {c: pd.DataFrame({"Timestamp": ts,
                               "Score": pred_per_channel[:, i].astype(np.uint8)})
              for i, c in enumerate(channels)}
    glob_scores = pd.DataFrame({"Timestamp": ts,
                                "Score": pred_per_channel.max(axis=1).astype(np.uint8)})

    out, ctx = {}, (contextlib.redirect_stdout(io.StringIO()) if quiet
                    else contextlib.nullcontext())
    with ctx:
        # --- aspects 1a/1b/3/5 : global ESAScores ---
        m = ESAScores(betas=BETA, select_labels=SEL)
        for k, v in m.score(y_true.drop(columns=["Channel"]), glob_scores).items():
            out[f"global_{k}"] = float(v)

        # --- aspects 2a/2b : channel- and subsystem-aware F0.5 ---
        sub_map = subsystems_mapping(ch_csv, channels)
        ca = ChannelAwareFScore(beta=BETA, select_labels=SEL)
        for k, v in ca.score(y_true.copy(), {c: d.copy() for c, d in y_pred.items()},
                             sub_map).items():
            out[f"chan_{k}"] = float(v)

        # --- aspect 4 : detection timing quality ---
        try:
            ad = ADTQC(select_labels=SEL)
            for k, v in ad.score(y_true.copy(), {c: d.copy() for c, d in y_pred.items()}).items():
                out[f"adtqc_{k}"] = float(v)
        except Exception as e:
            out["adtqc_error"] = str(e)[:120]

        # --- per-channel ESAScores (mean across channels) ---
        per = []
        for i, c in enumerate(channels):
            yt = y_true[y_true["Channel"] == c].drop(columns=["Channel"])
            if len(yt) == 0:
                continue
            sc = pd.DataFrame({"Timestamp": ts, "Score": pred_per_channel[:, i].astype(np.uint8)})
            try:
                per.append(ESAScores(betas=BETA, select_labels=SEL).score(yt, sc)[f"EW_F_{BETA:.2f}"])
            except Exception:
                pass
        if per:
            out["perchannel_EW_F_0.50_mean"] = float(np.mean(per))
    out["n_positive_rows"] = int(pred_per_channel.max(axis=1).sum())
    return out


PRIMARY = ["global_EW_F_0.50", "global_EW_precision", "global_EW_recall",
           "global_alarming_precision", "global_AFF_F_0.50",
           "chan_channel_F0.50", "chan_subsystem_F0.50",
           "adtqc_Total", "adtqc_AfterRate", "perchannel_EW_F_0.50_mean"]


def summarize(sb):
    keys = [k for k in PRIMARY if k in sb] + [k for k in sb if k not in PRIMARY]
    return " | ".join(f"{k.replace('global_','').replace('chan_','')}={sb[k]:.4f}"
                      if isinstance(sb[k], float) else f"{k}={sb[k]}" for k in keys)
