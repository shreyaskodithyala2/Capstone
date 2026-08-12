# Federated Anomaly Detection for Multi-Satellite Telemetry (ESA-ADB)

**Capstone Project 122 · PES University**
Shreyas Kodithyala (PES2UG23CS563) · Tulika Omer (PES2UG23CS657) ·
Veeramallu Meghana (PES2UG23CS684) · Vijay Raghav Reddy K (PES2UG23CS688)
Guide: Dr. Richa Sharma

---

## TL;DR — what this repo contains

We built spacecraft anomaly detectors on ESA's official **ESA-ADB** benchmark and **beat the
published state of the art on Mission 1**, plus a cross-mission federated learning framework.

| | Our best | Published best | |
|---|---|---|---|
| **Mission 1** event-wise F0.5 | **0.8532** | 0.786 *(Telemanom-ESA-Pruned)* | ✅ **+0.067** |
| **Mission 2** event-wise F0.5 | **0.9644** | 0.949 *(Windowed iForest)* | ✅ **+0.015** |
| **Mission 2** affiliation F0.5 | **0.9839** | 0.976 *(iForest)* | ✅ **+0.008** |

Measured with ESA's **own** metric code, on ESA's **own** train/test split, with **all tuning
confined to the training period**. Our scorer is validated to reproduce a published baseline
number exactly (0.9487 vs published 0.949).

### The finding that made it work

**ESA-ADB's standard preprocessing erases ~30% of Mission 1's anomalies.**

When no telemetry arrives, Zero-Order-Hold forward-fills the previous value, producing a
perfectly flat line that any forecaster predicts with near-zero error:

```
P(anomaly | no telemetry in a 30s window) = 56.7%     vs a 1.806% base rate  →  31× lift
30.7% of ALL Mission-1 anomalous rows live in these silent stretches
```

Recovering that signal with label-free **availability features** is the single biggest driver of
our result. This addresses ESA requirement **R8** (irregular time series), which the benchmark
authors state no algorithm satisfies. See [`docs/02-PREPROCESSING.md`](docs/02-PREPROCESSING.md).

---

## Where to start

| I want to… | Read |
|---|---|
| **Understand the dataset** | [`docs/01-DATASET.md`](docs/01-DATASET.md) |
| **Understand / rerun preprocessing** | [`docs/02-PREPROCESSING.md`](docs/02-PREPROCESSING.md) · [`preprocessing/README.md`](preprocessing/README.md) |
| **Understand the models** | [`docs/03-METHOD.md`](docs/03-METHOD.md) · [`models/README.md`](models/README.md) |
| **See every number** | [`docs/04-RESULTS.md`](docs/04-RESULTS.md) |
| **Understand ESA's 5-aspect metric suite** | [`docs/05-EVALUATION.md`](docs/05-EVALUATION.md) |
| **Know what's next** | [`docs/06-ROADMAP.md`](docs/06-ROADMAP.md) |
| **Know the literature (incl. competitors)** | [`docs/07-RELATED-WORK.md`](docs/07-RELATED-WORK.md) · [`papers/REFERENCES.md`](papers/REFERENCES.md) |
| **Continue this work with an AI assistant** | [`CLAUDE.md`](CLAUDE.md) — drop it in and go |

---

## Repository map

```
├── README.md                  ← you are here
├── CLAUDE.md                  ← complete portable context for resuming work
│
├── docs/                      ← all explanation lives here
│   ├── 01-DATASET.md            what ESA-ADB is, with real rows
│   ├── 02-PREPROCESSING.md      the pipeline + the ZOH/availability finding
│   ├── 03-METHOD.md             both models, step by step
│   ├── 04-RESULTS.md            every metric, every comparison
│   ├── 05-EVALUATION.md         ESA's 5 aspects and how we replicate them
│   ├── 06-ROADMAP.md            done / remaining / the research plan
│   └── 07-RELATED-WORK.md       incl. the SMED competitor
│
├── preprocessing/             ← reproduce the official ESA pipeline
│   ├── README.md                one-command setup + the 9 gotchas we hit
│   ├── setup_env.sh  run_preprocessing.sh  build_official_images.sh
│   ├── docker_build/            patched base image
│   └── patches/                 TimeEval GPU patch
│
├── models/                    ← ⭐ the two winning models
│   ├── README.md                which won what, and how they relate
│   ├── gdnv2/                   foundation + Mission-2 winner (0.9644)
│   └── gdnv4/                   Mission-1 winner (0.8532) + full metric suite
│
├── federated/                 ← cross-mission federated learning
│   └── src/                     FedProx + trimmed mean + personalization
│
├── logs/                      ← complete run audit trail
├── papers/                    ← 22 relevant PDFs + annotated bibliography
└── legacy/phase2-analysis/    ← earlier Phase-2 exploration (superseded)
```

**The datasets are not in this repo** (~8 GB raw, ~34 GB preprocessed).
Download links in [`preprocessing/README.md`](preprocessing/README.md).

---

## Quick start

```bash
# 1. code + data
git clone https://github.com/kplabs-pl/ESA-ADB
#    download ESA-AD from https://zenodo.org/records/15237121
#    unpack into ESA-Mission1/ and ESA-Mission2/

# 2. environment (handles 4 known packaging landmines)
bash preprocessing/setup_env.sh /path/to/ESA-ADB

# 3. preprocess (~4 h, ~34 GB, resumable, self-verifying)
bash preprocessing/run_preprocessing.sh /path/to/ESA-ADB /path/to/ESA-Mission1 /path/to/ESA-Mission2

# 4. availability features (the key ingredient) — reads the RAW pickles
conda activate timeeval
python models/gdnv2/src/extract_features.py

# 5. train + evaluate
python models/gdnv2/src/train.py --mission M1 --tag v1   # base model (~15 min, Apple MPS)
python models/gdnv4/src/run_v4.py --mission M1 --tag v4  # Mission-1 winner  → 0.8532
python models/gdnv2/src/m2_absolute.py                   # Mission-2 winner → 0.9644
```

> ⚠️ Paths inside the scripts are currently **absolute** (`/Users/Shreyas/...`).
> Edit the `ESA`, `PREP` and mission `raw` constants before running elsewhere.

---

## The two models — why there are two

| | **GDNv2** | **GDNv4** |
|---|---|---|
| Role | Foundation model + **Mission-2 winner** | **Mission-1 winner** |
| Score | M2 **0.9644** EW F0.5 | M1 **0.8532** EW F0.5 |
| Detection | EWMA(15) → robust z → absolute z*=20 | Telemanom prune + multi-span local z + gap detector + closing |
| Folder | [`models/gdnv2/`](models/gdnv2/) | [`models/gdnv4/`](models/gdnv4/) |

**They share one neural network.** GDNv4 is not a different model — it is a better *detection
stack* running on GDNv2's trained encoder. `models/gdnv4/src/` imports from `models/gdnv2/src/`.

**Why different detection stacks win on different missions** is itself a finding: Mission 1 needs
aggressive pruning (its anomalies are sharp and rare); Mission 2 is *destroyed* by pruning
(0.9644 → 0.014) because its rare events are subtle. This directly motivates *personalised
detection policies* in the federated setting.

---

## Honest status

**Verified:** preprocessing correct (reproduces a published number exactly) · Mission-1 result is
the best published anywhere · all five ESA aspects now measured.

**Not yet:** single seed only (3-seed stability run is mandatory before publication) ·
ADTQC regression on M1 (0.186 vs 0.901 — diagnosed, fix identified) · federation still runs on the
older weak model, not GDNv4 · full 76/100-channel sets untouched · Mission-2 v4 run in progress.

Full limitations in [`docs/04-RESULTS.md`](docs/04-RESULTS.md).

---

## Acknowledgements

Built on [ESA-ADB](https://github.com/kplabs-pl/ESA-ADB) by KP Labs, Airbus Defence and Space,
and the European Space Agency. Dataset: <https://zenodo.org/records/15237121>.
