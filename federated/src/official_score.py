"""Faithful replica of TimeEval experiments.py scoring so GDN/other models get numbers
DIRECTLY comparable to the official baselines (same y_true build + same ESAScores call).
Mirrors timeeval/core/experiments.py lines 339-355, 119, 154-155 exactly."""
import sys, contextlib, io
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/ESA-ADB")
from timeeval.metrics import ESAScores


def official_score(binary_scores, timestamps, labels_csv, anomaly_types_csv, target_channels,
                   select_labels=("Rare Event", "Anomaly"), betas=0.5):
    """binary_scores: 1D 0/1 per test timestamp (already the algorithm's max-over-channels output)."""
    ts = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    test_data_scores = pd.DataFrame({"Timestamp": ts, "Score": np.asarray(binary_scores).astype(np.uint8)})

    ldf = pd.read_csv(labels_csv, parse_dates=["StartTime", "EndTime"])
    ldf["StartTime"] = ldf["StartTime"].apply(lambda t: t.tz_localize(None) if t.tzinfo else t)
    ldf["EndTime"] = ldf["EndTime"].apply(lambda t: t.tz_localize(None) if t.tzinfo else t)
    # events fully inside the test range (experiments.py 347-348)
    ldf = ldf[ldf["StartTime"] >= test_data_scores["Timestamp"].min()]
    ldf = ldf[ldf["EndTime"] <= test_data_scores["Timestamp"].max()]
    # merge last-4 anomaly_types columns by ID (Category, Dimensionality, Locality, Length)
    at = pd.read_csv(anomaly_types_csv)
    cols = list(at.columns[-4:])
    for c in cols:
        ldf[c] = ""
    for _, row in at.iterrows():
        ldf.loc[ldf["ID"] == row["ID"], cols] = row[cols].values
    # keep target-channel rows, NOT deduped (experiments.py line 119)
    y_true = ldf[ldf["Channel"].isin(target_channels)]

    m = ESAScores(betas=betas, select_labels={"Category": list(select_labels)})
    with contextlib.redirect_stdout(io.StringIO()):        # silence metric debug prints
        return m.score(y_true.drop(columns=["Channel"]), test_data_scores)


if __name__ == "__main__":
    # validate against the official subsequence_if M2 output -> should reproduce 0.9487
    D = "/Users/Shreyas/Downloads/Capstone/ESA/ESA-ADB/results/2026_08_02_17_46_22/Subsequence IF/18a723dfc8c19b392599e660a78e8da8/ESA-Mission2/21_months/1"
    s = np.loadtxt(D + "/anomaly_scores.ts")
    ts = pd.read_csv("/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/data_light/multivariate/ESA-Mission2-semi-supervised/21_months.test.csv", usecols=["timestamp"])["timestamp"]
    res = official_score(s, ts,
                         "/Users/Shreyas/Downloads/Capstone/ESA/ESA-Mission2/labels.csv",
                         "/Users/Shreyas/Downloads/Capstone/ESA/ESA-Mission2/anomaly_types.csv",
                         [f"channel_{i}" for i in range(18, 29)])
    print("replica :", {k: round(v, 4) for k, v in res.items() if "EW" in k})
    print("official: EW_precision=0.9509 EW_recall=0.94 EW_F_0.50=0.9487")
