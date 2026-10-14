"""Project configuration: every folder path and every per-mission setting, in one place.

All other modules import from here, so moving the project means editing only this file
(or setting the ESA_* environment variables, which is how the Kaggle notebook redirects it).

Expected layout on the author's machine:
    ESA/                      <- ESA_ROOT
    |-- ESA-ADB/              official benchmark repo: preprocessed CSVs + timeeval metrics
    |-- ESA-Mission1/         raw ESA-ADB release: labels.csv, channels/*.zip, telecommands/*.zip
    |-- ESA-Mission2/
    `-- Main/                 <- this repository
        |-- data/             our availability features (+ data/full: 76/100-channel arrays)
        `-- results/          trained models and result JSONs
"""
import os

MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # this repository
ESA = os.environ.get("ESA_ROOT", os.path.dirname(MAIN))                # folder that holds Main/
PREP = os.environ.get("ESA_PREP", f"{ESA}/ESA-ADB/data/preprocessed/multivariate")  # official CSVs
DATA = os.environ.get("ESA_DATA", f"{MAIN}/data")                      # *_feats.npz, full/*.npy
RESULTS = os.environ.get("ESA_OUT", f"{MAIN}/results")                 # checkpoints + JSONs
CACHE = os.environ.get("ESA_CACHE", f"{RESULTS}/cache")                # full-set scratch (~9 GB)
TIMEEVAL_PATH = os.environ.get("TIMEEVAL_PATH", f"{ESA}/ESA-ADB")      # ESA's metric package

TC_TAU = 1800.0     # telecommand recency = exp(-seconds_since_last_command / 1800 s)

# One entry per experiment setting.
#   coll / train / test : official preprocessed CSV folder and file names
#   raw                 : raw ESA-ADB release folder (labels, channel + telecommand pickles)
#   chs                 : input channels, in column order
#   split / grid        : train/test boundary date and grid step in seconds (for reference)
#   sd_floor            : minimum std used when z-scoring a channel (see note below)
MISSIONS = {
    # Lightweight subsets: the 6 / 11 channels ESA-ADB uses for its headline results.
    "M1": dict(coll="ESA-Mission1-semi-supervised", raw=f"{ESA}/ESA-Mission1",
               chs=[f"channel_{i}" for i in range(41, 47)],
               train="84_months.train.csv", test="84_months.test.csv",
               split="2007-01-01", grid=30, sd_floor=0.0),
    "M2": dict(coll="ESA-Mission2-semi-supervised", raw=f"{ESA}/ESA-Mission2",
               chs=[f"channel_{i}" for i in range(18, 29)],
               train="21_months.train.csv", test="21_months.test.csv",
               split="2001-10-01", grid=18, sd_floor=0.0),
    # Full channel sets (ESA-ADB's default setting). Same CSVs, wider channel list.
    # The telecommand count does not change: only priority >= 3 commands are kept (11 / 4).
    "M1_full": dict(coll="ESA-Mission1-semi-supervised", raw=f"{ESA}/ESA-Mission1",
                    chs=[f"channel_{i}" for i in range(1, 77)],
                    train="84_months.train.csv", test="84_months.test.csv",
                    split="2007-01-01", grid=30, sd_floor=1e-3),
    "M2_full": dict(coll="ESA-Mission2-semi-supervised", raw=f"{ESA}/ESA-Mission2",
                    chs=[f"channel_{i}" for i in range(1, 101)],
                    train="21_months.train.csv", test="21_months.test.csv",
                    split="2001-10-01", grid=18, sd_floor=1e-3),
}
# Why sd_floor: the full sets contain near-constant channels (M1: 4 with sd ~1e-7, M2: 13 with
# sd < 1e-6). Dividing by such a tiny sd turns a 2.7e-4 blip into z ~ 2700, which then dominates
# the training loss. The floor leaves those channels flat instead. It is 0.0 for the lightweight
# subsets on purpose, so the published 0.8532 / 0.9644 stay bit-identical.
