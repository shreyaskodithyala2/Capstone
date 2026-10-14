"""Generate kaggle/m1_full_channels.ipynb: the notebook that runs Mission 1's full channel
set (76 channels) on a Kaggle GPU session -- train, tune on train, test once, report.

    conda activate timeeval
    python kaggle/export_kaggle_bundle.py     # -> kaggle/build/esa_meta.bundle + MANIFEST.json
    python kaggle/make_kaggle_notebook.py     # -> kaggle/m1_full_channels.ipynb

The notebook is pinned to the exact code bundle (BUNDLE_ID) and data files (FINGERPRINTS) it
was generated for, and refuses to run against anything else. Re-run both commands after ANY
change under src/, then upload the new esa_meta.bundle to Kaggle.

Each cell's source is a Python string below (SETTINGS, ENV_CHECK, LOCATE, ...); main() just
stitches them into an .ipynb file.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(HERE)
FULL = f"{MAIN}/data/full"
MK = "M1_full"


def fingerprints():
    """Cheap identity checks of the six data files: size, shape and a sampled sum (vals),
    first/last timestamp (ts), total sample count (feats). Embedded in the notebook."""
    fp = {}
    for sp in ("train", "test"):
        f = f"{FULL}/{MK}_{sp}_vals.npy"
        v = np.load(f, mmap_mode="r")
        fp[os.path.basename(f)] = dict(bytes=os.path.getsize(f), shape=list(v.shape),
                                       probe=round(float(np.asarray(v[::1009], np.float64).sum()), 3))
        f = f"{FULL}/{MK}_{sp}_ts.npy"
        t = np.load(f, mmap_mode="r")
        fp[os.path.basename(f)] = dict(bytes=os.path.getsize(f), shape=list(t.shape),
                                       first=str(t[0]), last=str(t[-1]))
        f = f"{FULL}/{MK}_{sp}_feats.npz"
        z = np.load(f)
        fp[os.path.basename(f)] = dict(bytes=os.path.getsize(f),
                                       count_sum=int(z["count"].sum(dtype=np.uint64)),
                                       tcdt_shape=list(z["tcdt"].shape))
    return fp


MD_INTRO = """# Anomaly detection on ESA-ADB Mission 1 — full channel set

**What this notebook does, in order**

1. **Train** the forecasting model (causal dilated TCN + graph attention) on the Mission 1
   training period (2000-01-01 → 2006-10-01; the last 3 months, → 2007-01-01, are the
   validation window, as in ESA-ADB §2.3): self-supervised next-step forecasting of all
   **76 channels**, with the **11 priority-3 telecommands** as input-only graph nodes.
   No labels are used for training.
2. **Tune** the four-detector stack on the **training** period's labels only, and write
   the winning configuration to `*_preregistered.json` *before* the test data is touched.
3. **Test once** (2007-01-01 → 2014-01-01) with that frozen configuration and report all
   five ESA aspects on the **58 `Target=YES` channels** — ESA-ADB's default full-set
   protocol, scored with ESA's own metric classes.

**Inputs:** the arrays dataset (six `M1_full_*` files) and the code dataset
(`esa_meta.bundle`). Paths are found automatically.

**Settings:** Accelerator **GPU T4 ×2** · Internet off is fine ·
**Save Version → Save & Run All (Commit)**.

**Where the time goes, and what runs where**

| stage | hardware | approx. |
|---|---|---|
| training | **both GPUs** (DataParallel), fp16 mixed precision if measured faster | ~22 min |
| model error arrays | GPU | ~5 min |
| z-scores, Telemanom pruning, thresholds | all CPU cores in parallel (thresholds on GPU) | ~20 min |
| TRAIN search, 305 configs | predictions on the **GPU**; ESA's metric on all CPU cores | ~4.6 h |
| TEST evaluation + 5-aspect scoreboard | all CPU cores | ~10 min |

**Measured total (first full run): 5.6 h.** ESA's metric classes and Telemanom's thresholding are CPU
code; they are run unmodified (exact ESA scores) and in parallel instead of being ported.
The run **stops itself** before Kaggle's 12 h limit, retries a failed stage with safer
settings, and never stops without writing `results/SUMMARY.md`.

**If anything still goes wrong:** attach that run's output as an input and run again. It is
detected automatically (`RESUME_DIR = "auto"`) and whatever is already done is skipped:
checkpoint, error arrays, and a pre-registered TRAIN config whose test never completed. With
the error arrays it can finish **without a GPU** (no GPU quota used).
"""

SETTINGS = '''# ---------------- settings ----------------
MISSION   = "M1_full"
BUNDLE_ID = "{bundle_id}"   # pinned: the attached esa_meta.bundle must be this exact one

# Validation = the last 3 months of the training period (ESA-ADB paper section 2.3;
# ESA's mission1_experiments.py splits Mission 1 at 2006-10-01).
TRAIN_ARGS  = ["--epochs", "30", "--bs", "24", "--seg", "512", "--patience", "4",
               "--val-from", "2006-10-01"]
DETECT_ARGS = ["--search", "two-stage"]
DETECT_WORKERS = "auto"    # CPU workers for pruning + search; "auto" sizes to cores and
                           # free RAM. Only lower it ("2", "1") after an out-of-memory kill.

# --- Kaggle limits: everything is measured from the moment this notebook starts ---
SESSION_LIMIT_H  = 12.0    # Kaggle kills a session at 12 h
SAFETY_MARGIN_H  = 0.75    # time kept for saving the output
TRAIN_BUDGET_H   = 4.0     # no training epoch may start that would end after this
TEST_RESERVE_MIN = 75      # the search stops early if needed to keep this for the test

KEEP_CACHES = False  # True: also save the 9 GB of error arrays to the output (slow to save;
                     # recomputing them on a resume takes ~5 min of GPU)
RESUME_DIR  = "auto" # "auto": if an earlier run's output (with M1_full_model.pt) is attached as an
                     # input, reuse whatever it has. None = always start fresh. Or give a path.
REUSE_PREREG = True  # reuse a saved TRAIN-only pre-registration whose test never completed
                     # (skips the ~2 h search; it is the same frozen config either way)
PREFLIGHT   = True   # ~3 min GPU / two-GPU check before the long training run

FINGERPRINTS = {fingerprints}
'''

ENV_CHECK = '''# ---------------- 1. environment ----------------
import os, sys, time, json, shutil, hashlib, subprocess, zipfile, resource, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

T_START = time.time()
DEADLINE = T_START + (SESSION_LIMIT_H - SAFETY_MARGIN_H) * 3600
def fail(msg):
    raise RuntimeError("\\n\\n!! " + msg + "\\n")
def hours_left():
    return (DEADLINE - time.time()) / 3600

# what an earlier run already produced (attached as an input dataset)
CACHE_FILES = [f"{MISSION}_{s}_{k}.npy" for s in ("train", "test") for k in ("ec", "valscache")]
PREREG_NAME = f"{MISSION}_preregistered.json"
if RESUME_DIR == "auto":
    found = []
    for dp, _, fs in os.walk("/kaggle/input", followlinks=True):
        if f"{MISSION}_model.pt" in fs:
            score = (PREREG_NAME in fs) + all(os.path.exists(f"{dp}/cache/{f}") for f in CACHE_FILES)
            found.append((score, dp))
    RESUME_DIR = max(found)[1] if found else None
    print("resume source:", RESUME_DIR or "none attached -- fresh run")
RES_CKPT = RES_CACHE = RES_PREREG = None
if RESUME_DIR:
    if os.path.exists(f"{RESUME_DIR}/{PREREG_NAME}"):
        RES_PREREG = f"{RESUME_DIR}/{PREREG_NAME}"
    if os.path.exists(f"{RESUME_DIR}/{MISSION}_model.pt"):
        RES_CKPT = f"{RESUME_DIR}/{MISSION}_model.pt"
    if all(os.path.exists(f"{RESUME_DIR}/cache/{f}") for f in CACHE_FILES):
        RES_CACHE = f"{RESUME_DIR}/cache"
    print("resume:", "checkpoint" if RES_CKPT else "-", "|", "error arrays" if RES_CACHE else "-",
          "|", "pre-registration" if RES_PREREG else "-")

print("python", sys.version.split()[0], "| numpy", np.__version__, "| pandas", pd.__version__,
      "| torch", torch.__version__, "| cpus", os.cpu_count())
GPU = torch.cuda.is_available()
if GPU:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name} | {p.total_memory/1e9:.1f} GB | sm_{p.major}{p.minor}")
    try:   # recent torch builds ship no kernels for Pascal (P100, sm_60)
        torch.nn.Conv1d(3, 8, 5).cuda()(torch.randn(2, 3, 64, device="cuda"))
        torch.cuda.synchronize()
    except Exception as e:
        fail(f"GPU present but unusable with this torch build ({str(e)[:120]}). "
             "Switch Accelerator to 'GPU T4 x2'.")
elif RES_CKPT and RES_CACHE:
    print("No GPU -- continuing on CPU: checkpoint and error arrays come from RESUME_DIR")
else:
    fail("No GPU. Settings -> Accelerator -> 'GPU T4 x2', then Save & Run All again. "
         "(A CPU-only session works only when RESUME_DIR holds the checkpoint AND cache/.)")
ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
print(f"RAM: {ram:.1f} GB" + ("   <- below 25 GB, the full set may not fit" if ram < 25 else ""))
print(f"hard stop at {time.strftime('%H:%M', time.localtime(DEADLINE))} "
      f"({hours_left():.2f} h from now)")
'''

LOCATE = '''# ---------------- 2. locate + verify inputs ----------------
# Works however Kaggle stored the uploads: unpacked or not, at any folder depth, under
# /kaggle/input/<slug>/ or /kaggle/input/datasets/<user>/<slug>/.
INPUT, WORK = "/kaggle/input", "/kaggle/working"

def walk_files(root=INPUT):
    for dp, _, fs in os.walk(root, followlinks=True):
        for f in fs:
            yield dp, f

def find_all(name, root=INPUT):
    return sorted({os.path.join(dp, f) for dp, f in walk_files(root) if f == name}, key=len)

def scratch_dir(need_gb):
    # a writable dir OUTSIDE /kaggle/working (whose contents become the output)
    for d in ("/kaggle/temp", "/tmp"):
        try:
            os.makedirs(d, exist_ok=True)
            if shutil.disk_usage(d).free / 1e9 >= need_gb:
                return d
        except OSError:
            pass
    fail(f"no scratch disk with {need_gb} GB free")

print("relevant files under", INPUT)
for dp, f in sorted(walk_files()):
    if f.endswith((".npy", ".npz", ".bundle", ".zip")) or f == "MANIFEST.json":
        print(f"  {os.path.getsize(os.path.join(dp, f))/1e6:10.1f} MB  {os.path.join(dp, f)}")

SCRATCH = scratch_dir(25)
print("scratch:", SCRATCH, f"({shutil.disk_usage(SCRATCH).free/1e9:.0f} GB free)")

# --- code bundle: esa_meta.bundle (Kaggle never unpacks it) or an unpacked meta/ / meta.zip.
# Every candidate is checked; the one whose bundle id matches this notebook is used.
cands = []
for i, z in enumerate(find_all("esa_meta.bundle") + find_all("meta.zip")):
    d = f"{SCRATCH}/meta_unz{i}"
    shutil.rmtree(d, ignore_errors=True)
    zipfile.ZipFile(z).extractall(d)
    cands += [(json.load(open(m))["bundle_id"], os.path.dirname(m)) for m in find_all("MANIFEST.json", d)]
cands += [(json.load(open(m))["bundle_id"], os.path.dirname(m)) for m in find_all("MANIFEST.json")]
match = [d for b, d in cands if b == BUNDLE_ID]
if not match:
    fail(f"code bundle {BUNDLE_ID} is not attached (bundles found: "
         f"{sorted({b for b, _ in cands}) or 'none'}). Upload the esa_meta.bundle generated "
         "with this notebook as a dataset and attach it (Add Input).")
META = f"{SCRATCH}/meta"                     # private, writable copy
shutil.rmtree(META, ignore_errors=True)
shutil.copytree(match[0], META)
man = json.load(open(f"{META}/MANIFEST.json"))
EMPTY = hashlib.sha256(b"").hexdigest()
missing, bad = [], []
for f, h in man["files"].items():
    p = f"{META}/{f}"
    if not os.path.exists(p):
        if h == EMPTY:                       # Kaggle drops empty files when it unpacks a zip
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").close()
        else:
            missing.append(f)
    elif hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
        bad.append(f)
if missing or bad:
    fail(f"code bundle damaged: missing {missing[:5]}, changed {bad[:5]}. "
         "Upload esa_meta.bundle (not a .zip) so Kaggle cannot unpack/alter it.")
print(f"code  OK: bundle {BUNDLE_ID}, {len(man['files'])} files verified (from {match[0]})")

# --- arrays: gathered into one scratch folder as symlinks. If Kaggle unpacked an .npz
# into a folder of .npy files, it is re-packed here.
DATA = f"{SCRATCH}/arrays"
shutil.rmtree(DATA, ignore_errors=True)
os.makedirs(DATA)
for split in ("train", "test"):
    for name in (f"{MISSION}_{split}_vals.npy", f"{MISSION}_{split}_ts.npy"):
        hits = find_all(name)
        if not hits:
            fail(f"{name} not found under {INPUT}. Attach the arrays dataset (Add Input).")
        os.symlink(hits[0], f"{DATA}/{name}")
    name = f"{MISSION}_{split}_feats.npz"
    hits = find_all(name)
    if hits:
        os.symlink(hits[0], f"{DATA}/{name}")
        continue
    parts = {f[:-4]: os.path.join(dp, f) for dp, f in walk_files()
             if os.path.basename(dp).startswith(f"{MISSION}_{split}_feats") and f.endswith(".npy")}
    if not {"count", "tsince", "tcdt"} <= set(parts):
        fail(f"{name} not found (neither the file nor an unpacked folder). "
             "Attach the arrays dataset with all six files.")
    print(f"re-packing {name} (Kaggle unpacked it) ...")
    np.savez(f"{DATA}/{name}", **{k: np.load(v) for k, v in parts.items()})

for f, exp in FINGERPRINTS.items():
    p = f"{DATA}/{f}"
    if f.endswith("_vals.npy"):
        if os.path.getsize(p) != exp["bytes"]:
            fail(f"{f}: {os.path.getsize(p):,} bytes, expected {exp['bytes']:,} (incomplete upload?)")
        v = np.load(p, mmap_mode="r")
        got = round(float(np.asarray(v[::1009], np.float64).sum()), 3)
        if list(v.shape) != exp["shape"] or abs(got - exp["probe"]) > 1e-3:
            fail(f"{f}: content differs from the files this notebook was built for")
    elif f.endswith("_ts.npy"):
        t = np.load(p, mmap_mode="r")
        if [str(t[0]), str(t[-1])] != [exp["first"], exp["last"]]:
            fail(f"{f}: timestamps differ")
    else:
        z = np.load(p)
        if (int(z["count"].sum(dtype=np.uint64)) != exp["count_sum"]
                or list(z["tcdt"].shape) != exp["tcdt_shape"]):
            fail(f"{f}: availability features differ")
        del z
print(f"arrays OK: {len(FINGERPRINTS)} files verified  ({DATA})")

# --- working layout
OUT, ROOT = f"{WORK}/results", f"{SCRATCH}/esa_root"
os.makedirs(OUT, exist_ok=True)
os.makedirs(ROOT, exist_ok=True)
if RES_CACHE:      # read-only input -> symlink into a writable scratch dir
    CACHE = f"{SCRATCH}/esa_cache"
    os.makedirs(CACHE, exist_ok=True)
    for f in CACHE_FILES:
        if not os.path.exists(f"{CACHE}/{f}"):
            os.symlink(f"{RES_CACHE}/{f}", f"{CACHE}/{f}")
    print("error arrays: reused from", RES_CACHE)
else:
    CACHE = f"{OUT}/cache" if KEEP_CACHES else f"{SCRATCH}/esa_cache"
    os.makedirs(CACHE, exist_ok=True)
if RES_PREREG and REUSE_PREREG and not os.path.exists(f"{OUT}/{PREREG_NAME}"):
    shutil.copy2(RES_PREREG, f"{OUT}/{PREREG_NAME}")
    print("pre-registration: reused from", RES_PREREG)
shutil.copytree(f"{META}/ESA-Mission1", f"{ROOT}/ESA-Mission1", dirs_exist_ok=True)  # labels
SRC = f"{META}/src"
ENV = dict(os.environ,
           ESA_ROOT=ROOT, ESA_DATA=DATA, ESA_OUT=OUT, ESA_CACHE=CACHE,
           TIMEEVAL_PATH=META,
           PYTHONPATH=os.pathsep.join([SRC, f"{META}/vendor", META]),
           PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
# Search: GPU + CPU workers (one thread each). Test: CPU only, CUDA hidden, so its fork
# workers start from a process with no CUDA state.
ENV_SEARCH = dict(ENV, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
ENV_TEST = dict(ENV, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1")

chk = subprocess.run([sys.executable, "-c",
    "import config, dataset, model, detectors, esa_metrics, telemanom_detector, gpu_search,"
    " parallel_scoring, run_full_channels, portion, more_itertools, torch;"
    "import timeeval.metrics as tm;"
    "print('imports OK | metrics:', tm.__file__);"
    "print('            portion:', portion.__file__);"
    "print('            mission channels:', len(config.MISSIONS['" + MISSION + "']['chs']),"
    "      '| cuda devices:', torch.cuda.device_count())"],
    cwd=SRC, env=ENV, capture_output=True, text=True)
print(chk.stdout.strip() or chk.stderr.strip()[-1500:])
if chk.returncode != 0:
    fail("subprocess import check failed (see above)")
'''

RUNNER = '''# ---------------- helper: run a script, keep its log; failures are RECORDED, not raised ----
# From here on nothing raises: a failed stage is retried (fewer workers / safer settings),
# and whatever happens is written to results/SUMMARY.md, so a committed run always finishes
# and always saves its output (checkpoint, pre-registration, logs) for a resume.
STATUS, STAGE_MIN = {}, {}

def run(script, args, log_name, env=None, stage=None):
    stage = stage or log_name
    t0 = time.time()
    log_path = f"{OUT}/{log_name}"
    with open(log_path, "w") as log:
        p = subprocess.Popen([sys.executable, "-u", script, *args], cwd=SRC, env=env or ENV,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        rc = p.wait()
    mins = (time.time() - t0) / 60
    STAGE_MIN[stage] = STAGE_MIN.get(stage, 0.0) + mins
    STATUS[stage] = "ok" if rc == 0 else (
        ("killed (out of memory)" if rc in (-9, 137) else f"failed (exit {rc})") + f" -- {log_path}")
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6   # KB -> GB on Linux
    print(f"\\n--- {script} [{stage}]: {STATUS[stage]} | {mins:.1f} min | peak RAM so far "
          f"{peak:.1f} GB | {hours_left():.2f} h left")
    return rc

def ladder(first):
    """worker counts to try: the setting, then 2, then 1 (each only once)"""
    out = []
    for w in (str(first), "2", "1"):
        if w not in out and (w == "auto" or first == "auto" or int(w) <= int(first)):
            out.append(w)
    return out
'''

PREFLIGHT_CELL = '''# ---------------- 3. preflight: GPUs, mixed precision, speed (~2 min) ----------------
CKPT = f"{OUT}/{MISSION}_model.pt"
if PREFLIGHT and GPU and not RES_CKPT and not os.path.exists(CKPT):
    pf = list(TRAIN_ARGS)
    pf[pf.index("--epochs") + 1] = "1"
    if run("train.py", ["--mission", MISSION, *pf, "--batches", "5",
                        "--ckpt", f"{OUT}/{MISSION}_preflight.pt"], "preflight.log",
           stage="preflight") == 0:
        if os.path.exists(f"{OUT}/{MISSION}_preflight.pt"):
            os.remove(f"{OUT}/{MISSION}_preflight.pt")
        lines = open(f"{OUT}/preflight.log").read().splitlines()
        steps = [l for l in lines if "-> AMP" in l or "-> fp32" in l]
        if steps:
            fast = float(steps[-1].split("AMP fp16")[1].split("ms")[0]) if "-> AMP" in steps[-1] \\
                else float(steps[-1].split("fp32")[1].split("ms")[0])
            ep_min = fast * 300 / 1000 / 60
            print(f"\\n~{fast:.0f} ms/step -> ~{ep_min:.1f} min/epoch -> at most "
                  f"~{ep_min * 30 / 60:.1f} h for 30 epochs (early stopping usually ends sooner)")
    else:
        print("preflight failed -- training will still be attempted (with a safe retry)")
else:
    print("preflight skipped")
'''

TRAIN = '''# ---------------- 4. train the encoder (both GPUs, mixed precision if faster) ----------------
if os.path.exists(CKPT):
    print("checkpoint already in", OUT, "-> training skipped")
elif RES_CKPT:
    shutil.copy2(RES_CKPT, CKPT)
    print("reusing checkpoint", RES_CKPT, "-> training skipped")
else:
    stop_by = min(T_START + TRAIN_BUDGET_H * 3600, DEADLINE - 2.0 * 3600)
    rc = run("train.py", ["--mission", MISSION, *TRAIN_ARGS, "--stop-by", str(stop_by)],
             "train.log", stage="train")
    if rc != 0 and not os.path.exists(CKPT):
        print("\\nretrying training with the safest settings (fp32, one GPU)")
        stop_by = min(time.time() + TRAIN_BUDGET_H * 3600, DEADLINE - 2.0 * 3600)
        run("train.py", ["--mission", MISSION, *TRAIN_ARGS, "--stop-by", str(stop_by),
                         "--amp", "off", "--dp", "off"], "train_retry.log", stage="train")
HAVE_CKPT = os.path.exists(CKPT)
print(f"checkpoint: {CKPT if HAVE_CKPT else 'NONE'} | {hours_left():.2f} h left")
'''

DETECT = '''# ---------------- 5. error arrays (GPU) -> TRAIN search (GPU + all CPU cores) -> TEST once ----
PREREG = f"{OUT}/{PREREG_NAME}"
RESULT = f"{OUT}/{MISSION}_result.json"
ok = HAVE_CKPT
# 5a  model error arrays for train + test (GPU, ~5 min); skipped when already cached
if ok:
    for attempt in (1, 2):
        rc = run("run_full_channels.py", ["--mission", MISSION, "--ckpt", CKPT, "--errors-only"],
                 "errors.log", stage="errors")
        if rc == 0:
            break
    ok = rc == 0
# 5b  TRAIN search: predictions assembled on the GPU, ESA's metric on every CPU core.
#     Writes the pre-registered config, then exits. Retried with fewer workers if killed.
if ok and not os.path.exists(PREREG):
    for w in ladder(DETECT_WORKERS):
        rc = run("run_full_channels.py", ["--mission", MISSION, "--ckpt", CKPT, *DETECT_ARGS,
                                    "--stage", "search", "--workers", w,
                                    "--deadline", str(DEADLINE),
                                    "--test-reserve-min", str(TEST_RESERVE_MIN)]
                 + (["--reuse-prereg"] if REUSE_PREREG else []),
                 f"search_w{w}.log", env=ENV_SEARCH, stage="search")
        if os.path.exists(PREREG) or rc not in (-9, 137):
            break
ok = ok and os.path.exists(PREREG)
# 5c  the single TEST evaluation of that frozen config (CPU, all cores, fresh process)
if ok:
    for w in ladder(DETECT_WORKERS):
        rc = run("run_full_channels.py", ["--mission", MISSION, "--ckpt", CKPT, *DETECT_ARGS,
                                    "--stage", "test", "--workers", w],
                 f"test_w{w}.log", env=ENV_TEST, stage="test")
        if rc == 0 or rc not in (-9, 137):
            break
print("\\nstage status:", json.dumps(STATUS, indent=1))
'''

RESULTS = '''# ---------------- 6. results vs ESA-ADB published full-set results ----------------
# ESA-ADB (TMLR version), Table 16, "Mission1 - trained and tested on the full set of
# channels": best value per metric over all published algorithms (TM = Telemanom-ESA).
PUB = [("Event-wise F0.5   (headline)", "global_EW_F_0.50",          0.061, "TM-Pruned"),
       ("Event-wise precision",         "global_EW_precision",       0.050, "TM-Pruned"),
       ("Event-wise recall",            "global_EW_recall",          0.967, "iForest"),
       ("Subsystem-aware F0.5",         "chan_subsystem_F0.50",      0.689, "TM"),
       ("Channel-aware F0.5",           "chan_channel_F0.50",        0.477, "TM"),
       ("Alarming precision",           "global_alarming_precision", 0.206, "TM-Pruned"),
       ("ADTQC score",                  "adtqc_Total",               0.752, "DC-VAE-ESA STD3"),
       ("Affiliation F0.5",             "global_AFF_F_0.50",         0.607, "TM-Pruned")]

lines = [f"# {MISSION} -- full channel set", "",
         f"* total notebook time: {(time.time()-T_START)/3600:.2f} h",
         "* stages: " + ", ".join(f"{k} {v:.0f} min ({STATUS.get(k, '?').split(' --')[0]})"
                                  for k, v in STAGE_MIN.items())]
if os.path.exists(RESULT) and os.path.exists(PREREG):
    pre, res = json.load(open(PREREG)), json.load(open(RESULT))
    sb, st = res["test"], res.get("search", {})
    lines += [f"* pre-registered config (chosen on TRAIN, F0.5 {pre['train_f05']:.4f}): "
              f"`{pre['config']}`",
              f"* search: {st.get('evaluated')} configs, {st.get('backend', 'cpu')} assembly"
              + ("  -- **truncated by the time budget**" if st.get("truncated") else ""),
              f"* inputs: {res['n_input_channels']} channels + telecommands; scored: "
              f"{len(res['scored_channels'])} Target=YES channels",
              f"* positive rows predicted: {sb.get('n_positive_rows'):,}", "",
              "| metric | ours | best published (full M1) |", "|---|---|---|"]
    for name, key, v, who in PUB:
        ours = sb.get(key)
        lines.append(f"| {name} | {ours:.4f} | {v:.3f} ({who}) |" if ours is not None
                     else f"| {name} | n/a | {v:.3f} ({who}) |")
    lines += ["", "Anomaly-only selection (ESA reports this separately):", ""]
    for k in ("anom_global_EW_F_0.50", "anom_global_EW_precision", "anom_global_EW_recall",
              "anom_chan_channel_F0.50", "anom_adtqc_Total"):
        if k in sb:
            lines.append(f"* {k}: {sb[k]:.4f}")
else:
    lines += ["", "**NOT FINISHED.** Stage status:", ""]
    lines += [f"* {k}: {v}" for k, v in STATUS.items()]
    lines += ["", "To continue: open this version's Output, create a dataset from "
              "`results/` (or add this notebook's output as an input), attach it, and run "
              "again. Everything already done (checkpoint, pre-registration) is reused "
              "automatically."]
summary = "\\n".join(lines)
open(f"{OUT}/SUMMARY.md", "w").write(summary + "\\n")
print(summary)
print("\\nfiles in", OUT)
for f in sorted(os.listdir(OUT)):
    pth = f"{OUT}/{f}"
    size = (sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(pth) for x in fs)
            if os.path.isdir(pth) else os.path.getsize(pth))
    print(f"  {f:40s} {size/1e6:9.1f} MB")
'''

MD_READ = """## Reading the result honestly

* **Comparable to:** ESA-ADB Table 16, *Mission1 – full set*: same inputs (all channels +
  priority-3 telecommands), same 58 scored channels, same labels, same metric classes,
  same 3-month validation window.
* **Not comparable to:** the lightweight-subset numbers (0.8532 / 0.786) — different
  channels, different events. ESA shows full-set scores collapse for every algorithm
  (Appendix D.2, false alarms grow with channel count), so expect far below 0.85.
* **Disclose:** single seed; detection settings chosen on train labels (allowed in
  ESA-ADB's semi-supervised setting); availability features (`count`, time-since-last-
  sample) come from raw timestamps the published baselines did not use; ESA's
  Telemanom-ESA trains on anomaly-free segments only, our encoder on all training rows.
"""


def cell(kind, src, i):
    """One notebook cell as the .ipynb JSON structure."""
    c = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {},
         "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main():
    """Fill in the bundle id + data fingerprints and write the notebook."""
    man = json.load(open(f"{HERE}/build/MANIFEST.json"))   # written by export_kaggle_bundle.py
    settings = SETTINGS.format(bundle_id=man["bundle_id"],
                               fingerprints=json.dumps(fingerprints(), indent=1))
    cells = [("markdown", MD_INTRO), ("code", settings), ("code", ENV_CHECK), ("code", LOCATE),
             ("code", RUNNER), ("code", PREFLIGHT_CELL), ("code", TRAIN), ("code", DETECT),
             ("code", RESULTS), ("markdown", MD_READ)]
    nb = {"cells": [cell(k, s, i) for i, (k, s) in enumerate(cells)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"},
                       "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                                  "isInternetEnabled": False, "language": "python",
                                  "sourceType": "notebook"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = f"{HERE}/m1_full_channels.ipynb"
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out}  (bundle {man['bundle_id']})")


if __name__ == "__main__":
    main()
