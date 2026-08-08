# Federated Learning for Anomaly Detection in Multi-Satellite Systems

**Capstone Project 122 · PES University · Phase 3**

Shreyas Kodithyala (PES2UG23CS563) · Tulika Omer (PES2UG23CS657) ·
Veeramallu Meghana (PES2UG23CS684) · Vijay Raghav Reddy K (PES2UG23CS688)
Guide: Dr. Richa Sharma

---

## What this is

We use ESA's official **ESA-ADB** benchmark — real telemetry from two operational spacecraft — to
build a **cross-mission federated Graph Deviation Network** for spacecraft anomaly detection.

Two satellites each train a local anomaly detector. They share only *model weights*, never raw
telemetry. Because Mission 1 and Mission 2 are genuinely different spacecraft with different
sensor counts (76 vs 100) and different physics, this is a realistic multi-satellite swarm — not a
simulation built by slicing one satellite's timeline.

### Headline results

Corrected event-wise F0.5 (the official ESA-ADB metric), lightweight subsets:

| | Mission 1 | Mission 2 |
|---|---|---|
| **Our federated GDN** | 0.337 | **0.936** |
| Published best baseline | 0.786 *(Telemanom-Pruned)* | 0.949 *(Windowed iForest)* |
| Published point detectors | ~0.0001 | — |

✅ **Pipeline verified** — our preprocessing reproduces the published Windowed-iForest score of
**0.949** to three decimals (we get **0.9487**).
✅ **Mission 2 is competitive** with the benchmark's best.
✅ **Federation with personalization improves both missions** over solo training.
❌ **Mission 1 is not there yet** — but we found the mechanism why (see below).

### The most interesting finding

**Standard Zero-Order-Hold preprocessing erases 30.7 % of Mission 1's anomalies.**

When no telemetry arrives in a window, ZOH forward-fills the previous value — producing a
perfectly flat line that any forecasting model predicts with near-zero error. Measured:

```
count = 0  →  P(anomaly) = 56.7 %   vs a 1.8 % base rate   (31× lift)
              and 30.7 % of ALL Mission-1 anomalies live here
```

This mechanically explains our Mission 1 recall of 0.09, and it is not reported anywhere in the
literature. Fixing it is contribution #1 (see [`docs/05-NEXT-STEPS.md`](docs/05-NEXT-STEPS.md)).

---

## Start here

| If you want to… | Read |
|---|---|
| Understand the dataset | [`docs/01-DATASET.md`](docs/01-DATASET.md) |
| Understand / run preprocessing | [`docs/02-PREPROCESSING.md`](docs/02-PREPROCESSING.md) · [`preprocessing/README.md`](preprocessing/README.md) |
| Understand the model | [`docs/03-GDN-MODEL.md`](docs/03-GDN-MODEL.md) |
| See all results | [`docs/04-RESULTS.md`](docs/04-RESULTS.md) |
| Know what to do next | [`docs/05-NEXT-STEPS.md`](docs/05-NEXT-STEPS.md) |
| Know what each paper says | [`papers/REFERENCES.md`](papers/REFERENCES.md) |
| Continue this work with an AI assistant | [`CLAUDE.md`](CLAUDE.md) |

---

## Quick start

```bash
# 1. get the code and data (data is NOT in this repo - ~8 GB)
git clone https://github.com/kplabs-pl/ESA-ADB
#    download ESA-AD from https://zenodo.org/records/15237121
#    unpack to ESA-Mission1/ and ESA-Mission2/

# 2. environment (conda; handles four known packaging landmines)
bash preprocessing/setup_env.sh /path/to/ESA-ADB

# 3. preprocess  (~4 h, ~34 GB output, resumable, self-verifying)
bash preprocessing/run_preprocessing.sh \
     /path/to/ESA-ADB /path/to/ESA-Mission1 /path/to/ESA-Mission2

# 4. train
conda activate timeeval
python src/train_m1.py                 # single mission
python src/federated.py                # solo vs federated comparison
python src/tune_best.py                # best comparable numbers
```

> Paths in `src/common.py` and `src/official_score.py` are currently **absolute**
> (`/Users/Shreyas/...`). Edit `PREP` and the mission `raw` paths before running elsewhere.

---

## Repository layout

```
├── README.md                   ← you are here
├── CLAUDE.md                   ← full context for resuming with an AI assistant
├── docs/
│   ├── 01-DATASET.md           what's in ESA-ADB, with real examples
│   ├── 02-PREPROCESSING.md     the pipeline, step by step + the ZOH finding
│   ├── 03-GDN-MODEL.md         GDN explained + our federated design
│   ├── 04-RESULTS.md           every number, with honest limitations
│   └── 05-NEXT-STEPS.md        prioritised roadmap
├── preprocessing/
│   ├── README.md               how to run it, and the 9 gotchas we hit
│   ├── setup_env.sh            one-command environment
│   ├── run_preprocessing.sh    one-command preprocessing + verification
│   ├── build_official_images.sh   build the Docker baselines locally
│   ├── docker_build/           patched base image (buster apt + exec bit fixes)
│   └── patches/                TimeEval GPU-request patch
├── src/
│   ├── gdn.py                  the model (federated-split aware)
│   ├── common.py               data loading, scoring
│   ├── federated.py            FedProx + trimmed mean + personalization
│   ├── official_score.py       ⭐ byte-faithful replica of official scoring
│   ├── telemanom_prune.py      faithful port of Telemanom thresholding+pruning
│   ├── train_m1.py  tune_best.py  tune_telemanom.py
│   ├── official_repro.py       run official baselines via Docker
│   └── make_light_csvs.py      channel-trimmed CSVs for memory-bound containers
├── results/                    raw run logs
└── papers/
    ├── REFERENCES.md           ⭐ annotated bibliography (incl. what to IGNORE)
    ├── benchmark/  models/  federated/  related/  phase2/
```

**The datasets are not in this repo** — ~8 GB compressed, ~34 GB preprocessed. Download links are
in [`preprocessing/README.md`](preprocessing/README.md).

---

## Key technical decisions

| Decision | Why |
|---|---|
| Use the **official** preprocessing, not a custom one | Comparability with published numbers is the foundation of every claim. Verified: reproduces 0.949 → 0.9487 |
| **Partial parameter sharing** for federation | M1 has 6 channels, M2 has 11 — you cannot average different-shaped embedding tables. Share dimension-independent operators; keep embeddings local |
| **Personalization after aggregation** | Without it, federation *hurt* Mission 1 (−0.040). With it, both missions improve. Independently confirmed by Atefrad & Karami |
| **Lightweight subsets first** | Full channel sets are a compute trap; the paper's citable numbers are on the subsets |
| **Mission 3 excluded** | The ESA-ADB authors exclude it from benchmarking |
| Keep **GDN** rather than switching to LSTM | LSTM *is* the SOTA baseline (Telemanom) — beating it with another LSTM proves nothing. GDN on ESA-ADB is unclaimed |

---

## Honest status

**What can be claimed:** verified preprocessing; GDN beating official point detectors on M1 by
orders of magnitude; competitive M2 (0.936 vs 0.949); federation helping both missions; a new
mechanistic explanation for M1 difficulty.

**What cannot:** beating Telemanom-Pruned on M1; a full official reproduction of the deep
baselines (needs GPU — their container is `tensorflow:2.5.1-gpu`); anything on the full
76/100-channel sets.

Full limitations in [`docs/04-RESULTS.md`](docs/04-RESULTS.md).

---

## Acknowledgements

Built on [ESA-ADB](https://github.com/kplabs-pl/ESA-ADB) by KP Labs, Airbus Defence and Space, and
ESA. Dataset: <https://zenodo.org/records/15237121>.
