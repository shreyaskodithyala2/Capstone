"""Package the code (and optionally the data arrays) needed for a full-set run on Kaggle.

Kaggle cannot hold the 23.5 GB of preprocessed CSVs or the raw pickles, so a run there uses:
  1. the arrays  : data/full/<mission>_{train,test}_{vals.npy, ts.npy, feats.npz}, uploaded
                   as one Kaggle dataset (already on disk; --arrays rebuilds them from the CSVs)
  2. the bundle  : kaggle/build/esa_meta.bundle, a zip (with a non-.zip name so Kaggle does
                   not unpack and alter it) holding
                     src/       the modules the notebook runs
                     timeeval/  ESA's metric package, metrics only (no docker / dask deps)
                     vendor/    portion, sortedcontainers, more_itertools (pinned versions)
                     ESA-Mission1/, ESA-Mission2/   labels.csv, anomaly_types.csv, ...
                     MANIFEST.json   sha256 of every file + a bundle id
     kaggle/build/MANIFEST.json is a copy, read by make_kaggle_notebook.py.

    python kaggle/export_kaggle_bundle.py                 # bundle only (after any src/ change)
    python kaggle/export_kaggle_bundle.py --arrays M1_full   # also rebuild data/full arrays
Then run kaggle/make_kaggle_notebook.py so the notebook pins the new bundle id.
"""
import hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile, time, zipfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC_DIR)
from config import MISSIONS, PREP, DATA, ESA                     # noqa: E402

BUILD = f"{HERE}/build"
# Everything run_full_channels.py and train.py import, and nothing else.
SRC = ["config.py", "dataset.py", "model.py", "train.py", "detectors.py",
       "telemanom_detector.py", "esa_metrics.py", "run_full_channels.py",
       "gpu_search.py", "parallel_scoring.py"]
META_CSV = ["labels.csv", "anomaly_types.csv", "channels.csv", "telecommands.csv"]
# Pure-Python scoring dependencies, copied from this environment: the versions under which
# our scorer reproduced the official harness exactly.
VENDOR = ["portion", "sortedcontainers", "more_itertools"]

# Replacement for timeeval/metrics/__init__.py: import only the three metric classes we use.
# The original imports every metric module, including one that breaks on NumPy 2.
MINIMAL_METRICS_INIT = '''"""Minimal metrics surface for the ESA-ADB scoreboard (ESAScores, ChannelAwareFScore,
ADTQC). The metric classes themselves are unchanged."""
from .metric import Metric
from .ranking_metrics import MultiChannelMetric, ChannelAwareFScore
from .latency_metrics import ADTQC
from .ESA_ADB_metrics import ESAScores

__all__ = ["Metric", "MultiChannelMetric", "ChannelAwareFScore", "ADTQC", "ESAScores"]
'''


def human(n):
    """Bytes -> readable size, e.g. 2252341248 -> '2.1 GB'."""
    for u in "B KB MB GB TB".split():
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def export_arrays(missions):
    """Rebuild data/full/<mission>_<split>_{vals.npy, ts.npy} from the official CSVs and copy
    the matching feats.npz next to them (the files dataset.py loads first)."""
    out = f"{DATA}/full"
    os.makedirs(out, exist_ok=True)
    for mk in missions:
        m = MISSIONS[mk]
        for split in ("train", "test"):
            csv = f"{PREP}/{m['coll']}/{m[split]}"
            t0 = time.time()
            v = pd.read_csv(csv, usecols=m["chs"], dtype=np.float32)[m["chs"]].to_numpy(np.float32)
            np.save(f"{out}/{mk}_{split}_vals.npy", v)
            print(f"  {mk} {split} vals {v.shape} {human(v.nbytes)} ({time.time()-t0:.0f}s)", flush=True)
            del v
            ts = pd.to_datetime(pd.read_csv(csv, usecols=["timestamp"])["timestamp"])
            np.save(f"{out}/{mk}_{split}_ts.npy", ts.to_numpy().astype("datetime64[ns]"))
            feats = f"{DATA}/{mk}_{split}_feats.npz"
            if not os.path.exists(feats):
                raise SystemExit(f"missing {feats} -- run src/extract_features.py {mk} first")
            shutil.copy2(feats, out)


def build_meta(meta, missions):
    """Fill the staging folder `meta` with everything the bundle contains."""
    # ESA's metrics package without tests/ and data/ (Kaggle would decompress its .gz files)
    shutil.copytree(f"{ESA}/ESA-ADB/timeeval/metrics", f"{meta}/timeeval/metrics",
                    ignore=shutil.ignore_patterns("__pycache__", "tests", "data", "*.gz"))
    with open(f"{meta}/timeeval/__init__.py", "w") as fh:
        fh.write('"""Metrics-only stand-in for the ESA-ADB timeeval package."""\n')
    with open(f"{meta}/timeeval/metrics/__init__.py", "w") as fh:
        fh.write(MINIMAL_METRICS_INIT)

    for mk in missions:                                        # label / channel metadata
        raw = MISSIONS[mk]["raw"]
        d = f"{meta}/{os.path.basename(raw)}"
        os.makedirs(d, exist_ok=True)
        for f in META_CSV:
            if os.path.exists(f"{raw}/{f}"):
                shutil.copy2(f"{raw}/{f}", d)

    os.makedirs(f"{meta}/src")
    for f in SRC:
        shutil.copy2(f"{SRC_DIR}/{f}", f"{meta}/src")

    for pkg in VENDOR:
        loc = importlib.util.find_spec(pkg).submodule_search_locations[0]
        shutil.copytree(loc, f"{meta}/vendor/{pkg}",
                        ignore=shutil.ignore_patterns("__pycache__", "tests"))


def import_check(meta):
    """Import the Kaggle entry points from the bundle alone (as on Kaggle) and fail if any
    local module would be picked up from outside it."""
    chk = subprocess.run([sys.executable, "-c",
                          "import sys; sys.path[:0]=sys.argv[1:]; import train, run_full_channels, "
                          "gpu_search, parallel_scoring; "
                          "bad=[m.__name__ for m in (train, run_full_channels, gpu_search, "
                          "parallel_scoring) if not m.__file__.startswith(sys.argv[1])]; "
                          "assert not bad, bad", os.path.abspath(f"{meta}/src")],
                         capture_output=True, text=True, cwd="/",
                         env=dict({k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                                  TIMEEVAL_PATH=meta))
    if chk.returncode != 0:
        raise SystemExit("bundle import check failed:\n" + chk.stderr[-1500:])


def write_manifest(meta):
    """sha256 of every file + bundle id (hash of all hashes). The notebook refuses to run if
    the attached bundle's id differs from the one it was generated for."""
    files = {}
    for r, _, fs in os.walk(meta):
        for f in fs:
            if "__pycache__" in r or f == "MANIFEST.json":
                continue
            fp = os.path.join(r, f)
            files[os.path.relpath(fp, meta)] = hashlib.sha256(open(fp, "rb").read()).hexdigest()
    bundle_id = hashlib.sha256(json.dumps(sorted(files.items())).encode()).hexdigest()[:12]
    json.dump({"bundle_id": bundle_id, "files": files}, open(f"{meta}/MANIFEST.json", "w"),
              indent=1, sort_keys=True)
    return bundle_id, len(files)


def main(missions, arrays=False):
    if arrays:
        export_arrays(missions)
    os.makedirs(BUILD, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        meta = f"{tmp}/meta"                      # the zip's top folder must be "meta/"
        build_meta(meta, missions)
        import_check(meta)
        bundle_id, n = write_manifest(meta)
        with zipfile.ZipFile(f"{BUILD}/esa_meta.bundle", "w", zipfile.ZIP_DEFLATED) as z:
            for r, _, fs in os.walk(meta):
                if "__pycache__" in r:
                    continue
                for f in fs:
                    fp = os.path.join(r, f)
                    z.write(fp, os.path.relpath(fp, tmp))
        shutil.copy2(f"{meta}/MANIFEST.json", f"{BUILD}/MANIFEST.json")
    print(f"  bundle_id {bundle_id}  ({n} files) -> {BUILD}/esa_meta.bundle "
          f"({human(os.path.getsize(f'{BUILD}/esa_meta.bundle'))})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--arrays"]
    main(args or ["M1_full", "M2_full"], arrays="--arrays" in sys.argv)
