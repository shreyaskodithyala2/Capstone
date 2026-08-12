# Preprocessing — run it end to end

Everything here reproduces the **official ESA-ADB preprocessing exactly**. We deliberately do
*not* use a custom pipeline, because comparability with the published benchmark numbers is the
foundation of every claim in this project.

> **Proof it works:** after running this, the official Windowed-iForest baseline scores
> **0.9487** corrected event-wise F0.5 on Mission 2 lightweight — the published paper reports
> **0.949**. Matching to three decimals is how we know the pipeline is correct.

---

## Step 0 — Get the data

The telemetry is **not** in this repo (it's ~8 GB compressed, ~34 GB preprocessed).

| Source | Link |
|---|---|
| **ESA-AD dataset (use this one)** | <https://zenodo.org/records/15237121> |
| Older 2-mission DOI (appears in some papers) | `10.5281/zenodo.12528695` |
| Official code repo | <https://github.com/kplabs-pl/ESA-ADB> |
| Project website | <https://ai4gs.space-codev.org/esa-adb> |

Download and unpack so you have this layout:

```
<somewhere>/
├── ESA-ADB/            # git clone https://github.com/kplabs-pl/ESA-ADB
├── ESA-Mission1/
│   ├── channels/           channel_1.zip … channel_76.zip   (pickled DataFrames)
│   ├── telecommands/       telecommand_*.zip                (698 files)
│   ├── channels.csv        telecommands.csv
│   ├── labels.csv          anomaly_types.csv
└── ESA-Mission2/
    ├── channels/           (100 files)
    ├── telecommands/       (123 files)
    └── channels.csv  telecommands.csv  labels.csv  anomaly_types.csv  events.csv
```

**Mission 3 is intentionally unused** — the ESA-ADB authors exclude it from benchmarking
(too many gaps, zero telecommands, trivial anomalies). Including it breaks comparability.

---

## Step 1 — Environment

```bash
bash preprocessing/setup_env.sh /path/to/ESA-ADB
```

Creates a conda env called `timeeval` and installs everything. It ends with a verification block
that must print `ALL CHECKS PASSED`.

### Four gotchas this script already handles for you

| Problem | Why it happens | Our fix |
|---|---|---|
| `No matching distribution found for requests====2.31.0` | ESA-ADB's `setup.py` has a typo — **four** equals signs | install with `--no-deps` |
| PEP-660 editable build hook fails | old setuptools contract | `--no-build-isolation` |
| `ModuleNotFoundError: natsort` | `infer_anomaly_types.py` needs it but it isn't declared anywhere | installed explicitly |
| numpy silently jumps to 2.x | installing torch pulls it, breaking the pandas/scipy ABI | torch installed `--no-deps`, numpy pinned at 1.21.6 and asserted |

---

## Step 2 — Preprocess

```bash
bash preprocessing/run_preprocessing.sh \
     /path/to/ESA-ADB /path/to/ESA-Mission1 /path/to/ESA-Mission2
```

- **Runtime** ~4 h total on an M4 MacBook Air (both missions)
- **Disk** ~34 GB of output
- **Safe to interrupt** — every write is guarded by an existence check, so re-running resumes
- **Backs up `anomaly_types.csv`** to `anomaly_types.orig.csv` before step 3 overwrites it
- Ends with an automatic verification pass

### What it runs, in order

1. `Mission1_semisupervised_prep_from_raw.py` → 5 train splits + shared test set
2. `Mission2_semiunsupervised_prep_from_raw.py` → 4 train splits + shared test set
3. `infer_anomaly_types.py` for both missions → regenerates the
   `Dimensionality` / `Locality` / `Length` columns in `anomaly_types.csv`

### Expected output

```
ESA-ADB/data/preprocessed/
├── datasets.csv                       ← index: 5 M1 rows + 4 M2 rows
└── multivariate/
    ├── ESA-Mission1-semi-supervised/
    │   ├── 3_months.train.csv    (245 MB)   10_months.train.csv  (824 MB)
    │   ├── 21_months.train.csv   (1.7 GB)   42_months.train.csv  (3.4 GB)
    │   ├── 84_months.train.csv   (6.8 GB)
    │   ├── 84_months.test.csv    (6.8 GB)   ← shared by all 5 splits
    │   └── *.metadata.json  (5)
    └── ESA-Mission2-semi-supervised/
        ├── 1_months.train.csv (242 MB)  5_months.train.csv  (1.2 GB)
        ├── 10_months.train.csv (2.4 GB) 21_months.train.csv (5.0 GB)
        ├── 21_months.test.csv  (4.9 GB) ← shared
        └── *.metadata.json  (4)
```

**Column layout** — Mission 1 = 175 columns, Mission 2 = 209 columns:

```
timestamp | channel_1 … channel_N | telecommand_… | is_anomaly_channel_… | is_anomaly_telecommand_…
```

Label values: `0` nominal · `1` anomaly · `2` rare nominal event · `3` communication gap.
(`is_anomaly_telecommand_*` columns are always 0 — `labels.csv` only annotates channels.)

See [`docs/02-PREPROCESSING.md`](../docs/02-PREPROCESSING.md) for a full walkthrough of *what*
the pipeline does to the data and *why*.

---

## Step 3 (optional) — Run the official baselines with Docker

Only needed if you want to reproduce the published baseline numbers yourself.

```bash
# 1. build the algorithm images locally (the HPI registry is private, so we build from source)
bash preprocessing/build_official_images.sh /path/to/ESA-ADB

# 2. apply the GPU patch (see below)
cd /path/to/ESA-ADB
git apply /path/to/this/repo/preprocessing/patches/timeeval-docker-nogpu.patch

# 3. make channel-trimmed CSVs so the containers fit in RAM
python src/make_light_csvs.py

# 4. run
python src/official_repro.py --mission M2 --algos subsequence_if
```

### Five problems you will hit, and the fixes (all included here)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `access forbidden` pulling images | `registry.gitlab.hpi.de/akita/i/*` is private | build locally — `build_official_images.sh` |
| 2 | `apt-get update` exits 100 | base image is Debian buster (EOL, repos archived) | patched base in `docker_build/python3-base/` drops the apt step |
| 3 | `exec /entrypoint.sh: permission denied` | `COPY` doesn't preserve the exec bit | patched Dockerfile adds `chmod +x` |
| 4 | container start → HTTP 500 | TimeEval requests a GPU for **every** container; Docker Desktop on macOS has none | `timeeval-docker-nogpu.patch` — set `TIMEEVAL_USE_GPU=1` to restore |
| 5 | container killed, exit 137 | full 175-column CSVs need ~10 GB; the Docker VM has ~7.7 GB | `make_light_csvs.py` trims to the subset channels (**result-identical** — the algorithms only ever read `target_channels`) |

**Build native for your architecture.** On Apple Silicon, building `linux/amd64` and running under
emulation works but is ~5–10× slower and long runs get killed. The script builds natively by default.

> **Telemanom-ESA and DC-VAE-ESA cannot practically run on a Mac** — their base image is
> `tensorflow:2.5.1-gpu` (amd64, CUDA). Use Kaggle/Colab for those.
