"""Proves our scorer (src/esa_metrics.py) gives the same numbers as ESA's official harness.

We take an anomaly-score file produced by the OFFICIAL ESA-ADB Docker pipeline (Windowed
iForest on Mission 2 lightweight, published F0.5 = 0.949), score it with our code, and
compare every headline metric with the metrics.csv the official harness wrote
(EW precision 0.9509, recall 0.94, F0.5 0.9487, ...), to full float precision.

    python tests/test_scorer_matches_official.py

Needs the official run output under ESA-ADB/results/ (produced once with Docker).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import ESA, DATA, MISSIONS
from esa_metrics import global_f05

OFFICIAL_RUN = (f"{ESA}/ESA-ADB/results/2026_08_02_17_46_22/Subsequence IF/"
                "18a723dfc8c19b392599e660a78e8da8/ESA-Mission2/21_months/1")
KEYS = ["alarming_precision", "EW_precision", "EW_recall", "EW_F_0.50",
        "AFF_precision", "AFF_recall", "AFF_F_0.50"]


def main():
    scores = np.loadtxt(f"{OFFICIAL_RUN}/anomaly_scores.ts")          # official 0/1 output
    official = pd.read_csv(f"{OFFICIAL_RUN}/metrics.csv").iloc[0]     # official harness scores
    ts = pd.Series(np.load(f"{DATA}/full/M2_full_test_ts.npy"))      # same test grid
    assert len(scores) == len(ts), (len(scores), len(ts))
    m = MISSIONS["M2"]
    r = global_f05(scores[:, None], ts, m["raw"], m["chs"])
    for k in KEYS:
        ours, theirs = r[f"global_{k}"], float(official[f"Rare Event_Anomaly_{k}"])
        assert abs(ours - theirs) < 1e-12, (k, ours, theirs)
        print(f"  OK  {k:18s} ours {ours:.6f} == official {theirs:.6f}")
    print("\nSCORER MATCHES THE OFFICIAL HARNESS")


if __name__ == "__main__":
    main()
