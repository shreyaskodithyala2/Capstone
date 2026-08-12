"""Create channel-trimmed copies of the preprocessed CSVs (lightweight subset only).
The official algorithms use only target_channels, so trimming other channels yields
IDENTICAL results while cutting memory ~15x so it fits a 16GB laptop's Docker VM."""
import os, shutil, time
import pandas as pd

SRC = "/Users/Shreyas/Downloads/Capstone/ESA/ESA-ADB/data/preprocessed"
DST = "/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/data_light"

JOBS = [
    ("ESA-Mission1-semi-supervised", [f"channel_{i}" for i in range(41, 47)], ["84_months.train.csv", "84_months.test.csv"]),
    ("ESA-Mission2-semi-supervised", [f"channel_{i}" for i in range(18, 29)], ["21_months.train.csv", "21_months.test.csv"]),
]

os.makedirs(DST, exist_ok=True)
shutil.copy(os.path.join(SRC, "datasets.csv"), os.path.join(DST, "datasets.csv"))
print("copied datasets.csv")

for coll, chs, files in JOBS:
    outdir = os.path.join(DST, "multivariate", coll)
    os.makedirs(outdir, exist_ok=True)
    keep = ["timestamp"] + chs + [f"is_anomaly_{c}" for c in chs]
    for f in files:
        t0 = time.time()
        src = os.path.join(SRC, "multivariate", coll, f)
        dst = os.path.join(outdir, f)
        # read only needed columns (memory-light), write trimmed
        df = pd.read_csv(src, usecols=keep)
        df = df[keep]  # enforce column order
        df.to_csv(dst, index=False, lineterminator="\n")
        # copy metadata json alongside (harness may look for it)
        for ext in [".metadata.json"]:
            mf = src.replace(".train.csv", ext).replace(".test.csv", ext)
            if os.path.exists(mf):
                shutil.copy(mf, os.path.join(outdir, os.path.basename(mf)))
        print(f"  {coll}/{f}: {df.shape} -> {os.path.getsize(dst)/1e6:.0f} MB | {time.time()-t0:.0f}s", flush=True)

print("DONE")
