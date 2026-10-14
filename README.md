# Federated Anomaly Detection for Multi-Satellite Telemetry (ESA-ADB)

**Capstone Project 122, PES University** — *Federated Learning Architecture for Health
Monitoring and Anomaly Detection in Multi-Satellite Systems.*

Satellite operators need alarms that are **precise** (every false alarm costs an engineer's
time) and that work **across a fleet** without pooling every spacecraft's raw telemetry in one
place. We build a graph-based forecasting model, a four-detector alarm stack on top of it, and a
federated setup in which two real spacecraft train together while keeping their data local.
Everything is evaluated with ESA's official benchmark, **ESA-ADB** (real, anonymised telemetry
from two ESA missions), using ESA's own metric code.

---

## Results

Corrected event-wise **F0.5** (ESA's headline metric: event-level, precision weighted 4x,
penalises the duration of false alarms). All settings were chosen on the training period;
the test period was evaluated once.

| ESA-ADB setting | ours | best published |
|---|---|---|
| Mission 1, lightweight (6 channels) | **0.8532** (precision 0.999) | 0.786 Telemanom-ESA · 0.801 SMED |
| Mission 2, lightweight (11 channels) | **0.9644** | 0.949 Windowed iForest · 0.971 SMED |
| Mission 1, full set (76 in / 58 scored) | **0.4394** | 0.061 Telemanom-ESA-Pruned |

All five ESA evaluation aspects, Mission 1:

| aspect | lightweight | published best | full set | published best |
|---|---|---|---|---|
| event-wise F0.5 | **0.8532** | 0.786 | **0.4394** | 0.061 |
| subsystem-aware F0.5 | 0.5385 | — | 0.3429 | 0.689 |
| channel-aware F0.5 | 0.4740 | 0.837 | 0.1806 | 0.477 |
| alarming precision | 0.7955 | 0.875 | **0.8085** | 0.206 |
| ADTQC (alarm timing) | 0.1864 | 0.901 | 0.4650 | 0.752 |
| affiliation F0.5 | 0.6812 | 0.734 | 0.5870 | 0.607 |

Channel attribution and alarm timing are the open weaknesses.

**Federated prototype** (two missions as two clients, earlier window model, val-tuned threshold):
Mission 2 improves from 0.923 (solo) to 0.936 (federated + personalised); Mission 1 drops
(0.080 -> 0.018). Moving the federated loop onto the current model is in progress.

---

## What was built, end to end

1. **Official preprocessing** (`preprocessing/`): ESA-ADB's own pipeline resamples every channel
   onto a fixed grid (30 s / 18 s). We reproduced the official baselines with ESA's Docker
   harness (Windowed iForest on Mission 2: 0.9487 vs the published 0.949) to prove the setup.
2. **Key finding — the grid hides missing data.** When no telemetry arrives, the official
   resampling copies the last value forward, so a forecaster sees a flat line and predicts it
   perfectly. On Mission 1, grid slots with no real sample are anomalous **56.7 %** of the time
   (31x the base rate, ~30 % of all anomalies). `src/extract_features.py` recovers this from the
   raw timestamps as label-free features: `count` (samples per slot), `tsince` (seconds since the
   last sample) and `tcdt` (seconds since each priority-3 telecommand).
3. **Forecasting model** (`src/model.py`, 38k parameters): every channel and every telecommand is
   a graph node with 3 inputs per step. A causal dilated **TCN** summarises the last 253 steps
   (~2 h) of each node; a **Graph Deviation Network** lets each channel attend to its most
   similar channels and to the telecommands; a small MLP predicts the next value. The anomaly
   signal is the forecast error. Trained self-supervised (`src/train.py`), no labels.
4. **Detection stack** (`src/detectors.py`, `src/telemanom_detector.py`), per channel:
   A) Telemanom dynamic threshold + pruning (the precision engine), B) multi-scale robust
   z-score, C) "telemetry silent too long", D) a lower threshold shortly after a telecommand;
   OR-ed, then nearby alarms merged and blips dropped. Mission 2 needs a simpler rule
   (smoothed robust z over an absolute threshold): the same encoder, a personalised decision rule.
5. **Evaluation** (`src/esa_metrics.py`): ESA's metric classes with the benchmark harness's exact
   input preparation; verified to reproduce ESA's official scores to 1e-12.
6. **Scale-up** (`src/run_full_channels.py`, `kaggle/`): the full 76-channel Mission 1 on a Kaggle
   GPU, with fixes that only matter at scale (per-channel silence lengths, near-constant
   channels, sparse graph, scoring only ESA's target channels, per-channel thresholds).
7. **Federated learning** (`federated/`): FedProx local training, only the shared weights
   (TCN/attention/head) leave a client, trimmed-mean aggregation, then per-mission fine-tuning.
   Embedding tables stay local, so missions with different channel counts can federate.

---

## Repository

```
src/                     the pipeline
  config.py              all paths and per-mission settings
  extract_features.py    raw pickles -> availability features (data/*_feats.npz)
  dataset.py             model input X (time, nodes, 3)
  model.py               GDNForecaster: TCN + graph attention + head
  train.py               training -> results/<mission>_model.pt
  detectors.py           detectors B, C, gap cleanup, alarm merging
  telemanom_detector.py  detector A (Telemanom threshold + pruning)
  esa_metrics.py         ESA's official metrics, all five aspects
  run_m1_lightweight.py  Mission 1 result (0.8532)
  run_m2_lightweight.py  Mission 2 result (0.9644)
  run_full_channels.py   full channel sets (Kaggle); gpu_search.py, parallel_scoring.py speed it up
federated/               federated-learning prototype (two missions = two clients)
preprocessing/           environment setup, official preprocessing, official baselines (Docker)
kaggle/                  builds the Kaggle code bundle + notebook for the full-set run
tests/                   exactness, scorer-vs-official, and result-reproduction checks
results/                 trained models + result files of every reported number
logs/                    run logs (full-set Kaggle run, federated prototype, reproduction)
docs/ARCHITECTURE.md     every stage in detail, with tensor shapes
```

## Run it

The ESA-ADB data is not in this repo. Download it from <https://zenodo.org/records/15237121>,
clone <https://github.com/kplabs-pl/ESA-ADB>, and place `ESA-ADB/`, `ESA-Mission1/`,
`ESA-Mission2/` next to this repository (paths are set in `src/config.py`).

```bash
bash preprocessing/setup_env.sh ../ESA-ADB          # conda env "timeeval"
bash preprocessing/run_preprocessing.sh ../ESA-ADB ../ESA-Mission1 ../ESA-Mission2
conda activate timeeval

python src/extract_features.py                      # availability features
python src/train.py --mission M1                    # or use results/M1_model.pt
python src/run_m1_lightweight.py                    # -> results/M1_result.json
python src/train.py --mission M2
python src/run_m2_lightweight.py                    # -> results/M2_result.json
python federated/run_federated.py                   # federated prototype

python tests/test_detectors_exact.py                # fast optimisations == original code
python tests/test_scorer_matches_official.py        # our scorer == ESA's official harness
python tests/test_reproduce_results.py              # recomputes 0.8532 and 0.9644 (~11 min)
```

Full channel set: `python kaggle/export_kaggle_bundle.py`, then
`python kaggle/make_kaggle_notebook.py`; upload the bundle and the `M1_full_*` arrays to Kaggle
and run `kaggle/m1_full_channels.ipynb` on a T4 x2 GPU session (~5.6 h).

## Limitations

- Single training seed; the Mission 2 margin over the published best is about 1–2 events.
- Our availability features use raw timestamps that the published baselines did not use.
- Scores come from ESA's metric code via a validated replica of the harness, not the full
  Docker pipeline.
- The federated loop has not yet been run on the current model.
