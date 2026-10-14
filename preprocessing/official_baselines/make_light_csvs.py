"""Channel-trimmed copies of the official preprocessed CSVs, for running the baselines in Docker.

The full CSVs (175 / 209 columns, ~10 GB) do not fit a laptop's Docker VM (exit 137, OOM).
The baselines only read the lightweight target channels, so keeping just
`timestamp | target channels | their is_anomaly columns` gives IDENTICAL results.

    python preprocessing/official_baselines/make_light_csvs.py
Output: <ESA_ROOT>/data_light/ (same folder layout as ESA-ADB/data/preprocessed/)
"""
import os, sys, shutil, time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from config import ESA, MISSIONS                                  # noqa: E402

SRC = f"{ESA}/ESA-ADB/data/preprocessed"
DST = f"{ESA}/data_light"


def main():
    os.makedirs(DST, exist_ok=True)
    shutil.copy(f"{SRC}/datasets.csv", f"{DST}/datasets.csv")     # TimeEval's dataset index
    for mk in ("M1", "M2"):
        m = MISSIONS[mk]
        outdir = f"{DST}/multivariate/{m['coll']}"
        os.makedirs(outdir, exist_ok=True)
        keep = ["timestamp"] + m["chs"] + [f"is_anomaly_{c}" for c in m["chs"]]
        for f in (m["train"], m["test"]):
            t0 = time.time()
            src = f"{SRC}/multivariate/{m['coll']}/{f}"
            df = pd.read_csv(src, usecols=keep)[keep]               # needed columns, fixed order
            df.to_csv(f"{outdir}/{f}", index=False, lineterminator="\n")
            meta = src.replace(".train.csv", ".metadata.json").replace(".test.csv", ".metadata.json")
            if os.path.exists(meta):                                # the harness reads it
                shutil.copy(meta, outdir)
            print(f"  {m['coll']}/{f}: {df.shape} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
