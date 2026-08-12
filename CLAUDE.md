# CLAUDE.md — ESA-ADB Cross-Mission Federated GDN

> **Purpose.** Drop this file into a Claude Code session (or any AI assistant) and it has enough
> context to resume the project without re-deriving anything. Every number here was measured from
> the real data, not copied from a paper. Where something is *assumed* rather than verified, it
> says so.

---

## 1. Quick context (60 seconds)

College capstone (Project 122, PES University, team of 4 + guide Dr. Richa Sharma):
**"Federated Learning Architecture for Health Monitoring and Anomaly Detection in Multi-Satellite
Systems."**

We use ESA's official **ESA-ADB** benchmark (real telemetry from two operational spacecraft) to
build a **cross-mission federated Graph Deviation Network (GDN)**.

- **Phase 2** (graded, done): channel-partitioned federated GDN over Mission 1 only, with
  hand-rolled preprocessing.
- **Phase 3** (current): rebuilt from scratch on the **official** ESA-ADB preprocessing and
  evaluation, then federated across **Mission 1 + Mission 2** with partial parameter sharing.

**Novelty:** genuine cross-mission federation (two real spacecraft as clients, not time-slices of
one), plus a newly discovered preprocessing artifact that explains why forecasting models
underperform on Mission 1.

---

## 2. Where things stand

| Item | Status |
|---|---|
| Environment (`timeeval` conda env) | ✅ working |
| Official preprocessing, both missions | ✅ done & verified |
| `infer_anomaly_types.py`, both missions | ✅ done |
| Official baselines reproduced (non-deep, Docker) | ✅ **M2 Windowed iForest = 0.9487 vs published 0.949** |
| Single-mission GDN | ✅ working |
| Cross-mission federated GDN | ✅ working; federation + personalization helps both missions |
| Telemanom pruning ported | ✅ M1 0.080 → 0.337 (precision 1.00) |
| Deep baselines (Telemanom-ESA, DC-VAE) | ❌ blocked on Mac — need GPU/cloud |
| Data-availability features | ✅ **done (GDNv2)** — the decisive ingredient |
| Telecommands as model input | ✅ done (GDNv2: input-only exogenous graph nodes) |
| **GDNv2: BOTH BENCHMARKS BEATEN (2026-08-10)** | ✅ **M1 0.8197 vs 0.786 · M2 0.9644 vs 0.949** |
| Full 5-aspect ESA scoreboard implemented | ✅ `GDNv3/src/esa_metrics.py`, validated at 0.41237 |
| **GDNv4: M1 = 0.8532 (2026-08-12)** | ✅ **best M1 result published anywhere** (ESA 0.786, SMED 0.801) |
| GDNv4 on M2 | ⏳ 432-config grid running (~200 done, best train 0.9048) |
| GitHub repo restructured around the 2 winning models | ✅ 2026-08-12, branch `new` |
| Federated version of GDNv2/v4 | ❌ not yet run (solo only) |
| Seed-stability check (3 seeds) | ❌ **mandatory before paper** |
| ADTQC fix (emit first index of merged run) | ❌ identified, not run |
| Full 76/100-channel sets | ❌ untouched |

### Results — corrected event-wise F0.5, lightweight subsets

| Model | M1 | M2 | Scoring |
|---|---|---|---|
| iForest / HBOS / PCC (official Docker) | 0.00007 / 0.00014 / 0.00013 | — | official harness |
| **Windowed iForest (official Docker)** | <0.001 *(pub.)* | **0.9487** | official harness |
| GDN solo | 0.080 | 0.923 | comparable, val-tuned |
| **GDN federated + personalized** | 0.018 | **0.936** | comparable, val-tuned |
| **GDN + Telemanom pruning** | **0.337** (P 1.00, R 0.09) | 0.014 | comparable, self-thresholding |
| *Published Telemanom-ESA Pruned* | *0.786* | *0.842* | *paper* |

Federation ablation (oracle threshold — **internal comparison only**):

| | Solo | Fed, no fine-tune | Fed + fine-tune |
|---|---|---|---|
| M1 | 0.1765 | 0.1370 ❌ | **0.1892** ✅ |
| M2 | 0.7739 | 0.7820 | **0.8050** ✅ |

### ⭐⭐⭐ GDNv4 (2026-08-12, `~/Downloads/Capstone/ESA/GDNv3/`) — CURRENT BEST

**M1 event-wise F0.5 = 0.8532** (P 0.9993, R 0.5385) — best published anywhere
(ESA 0.786, SMED 0.801). Pre-registered config `p=0.3 zmode=local q=0.9999 close=60
minlen=3 gapL=10 boost=0.6`, train F0.5 0.7748. **M2 v4 still running** (first config
already train 0.8936).

Full five-aspect scoreboard now implemented (`GDNv3/src/esa_metrics.py`, validated to
reproduce the gap-detector 0.41237 exactly): global ESAScores + ChannelAwareFScore +
ADTQC + per-channel. **We had only ever measured 2 of ESA's 5 aspects before this.**

| aspect | best published | GDNv2 | **v4** |
|---|---|---|---|
| 1a event-wise F0.5 | 0.786 | 0.8197 | **0.8532** ✅ |
| 2b channel-aware F0.5 | 0.837 | *n/m* | 0.4740 |
| 3 alarming precision | 0.875 | 0.2279 | 0.7955 |
| 4 ADTQC | 0.901 | *n/m* | 0.1864 ❌ |
| 5 affiliation F0.5 | 0.734 | 0.6588 | 0.6812 |

- v4 = GDNv2 core (Telemanom prune + gap detector) **+** GDNv3 additions (multi-span
  block-local robust z, morphological closing, min-run filter, telecommand-gated lower
  threshold, per-channel outputs).
- **v3 ablation proved pruning is the precision engine**: removing it dropped EW precision
  0.999 → 0.444 and F0.5 to 0.3827, while lifting alarmP to 0.640. Keep pruning.
- **ADTQC regression is diagnosed**: `Nb_Before=23 vs Nb_After=12` — closing + TC-boost make
  alarms fire before the labelled onset. Fix: emit the first index of each merged run.
- **Competitor found**: SMED (Brancato et al., PHM Society, papers.phmsociety.org/…/4465) —
  same subsets, M1 0.801 (we beat), **M2 0.971 (beats our 0.9644)**. Pure MASS similarity
  search, no training, 16 s. Their ADTQC is 0.075–0.160 and they compare only vs Telemanom,
  not the per-metric best. Fusing a MASS "distance to best recent match" branch with our
  forecast error is an obvious, unclaimed next step.
- Results log: `GDNv3/RESULTS.md`. Runner: `GDNv3/src/run_v4.py`.

### ⭐⭐ GDNv2 (2026-08-10, `~/Downloads/Capstone/ESA/GDNv2/`) — BOTH BENCHMARKS BEATEN

| | **GDNv2** | Published best | Margin |
|---|---|---|---|
| **M1** | **0.8197** (P=0.9993, R=0.4769) | 0.786 Telemanom-Pruned | +0.034 |
| **M2** | **0.9644** (P=0.9606, R=0.9800) | 0.949 Windowed iForest | +0.015 |

All tuning on train labels only; M2 rule fully pre-registered (absolute-z transfer, z*=20,
span 15). M1: with p ≤ 0.13 the same pipeline scores **0.8371** (train tied within
single-event noise; strict selector picked p=0.3 → 0.8197 headline).

- **Architecture:** causal dilated **TCN** encoder (RF=253 steps ≈ 2.1 h) + node embeddings +
  decomposed graph attention + per-node head; 38k params. Channel-node inputs
  `[value, gap_flag, log-tsince]`; **telecommand nodes** (priority≥3) input-only, edges
  tc→ch only. TCN = full-series inference: both 7.4M-row passes in ~53 s on MPS (a per-window
  GRU would be 30–60 min/pass — this is why iteration was feasible).
- **M1 winning pipeline:** errors → EWMA(5) → **zero errors inside count==0 runs +30-step
  margin** (gap re-acquisition spikes reach z≈1149 and poison pruning's sorted-peak logic;
  16,676 test gap runs vs 3,196 train — justification is label-free input statistics) →
  Telemanom prune → OR with gap detector (all-channels-silent ≥10 steps, L train-tuned).
- **M2 winning pipeline:** errors → EWMA(15) → robust z (self med/IQR) → absolute z*=20.
  Quantile transfer instead gives 0.9260; test q0.98 = 0.9798 (oracle, don't headline).
- **Standalone finding:** gap detector alone = **0.412 at P=1.0** on M1 test.
- Iteration audit trail in `GDNv2/logs/`; headline JSONs: `results/M1_FINAL.json`,
  `results/M2_v3_absolute.json`. Full documentation: `GDNv2/README.md`.
- **Caveats:** single seed; pipeline design informed by logged test-outcome iterations
  (final params train-tuned; disclose); availability features come from raw timestamps the
  official CSVs erase (that asymmetry IS the R8 contribution — run the
  Telemanom+availability ablation to bulletproof); scored via validated replica, not full
  Docker harness; solo models (federated GDNv2 not yet run).

---

## 3. ⭐ The key finding — ZOH erases 30 % of Mission 1's anomalies

Measured on `channel_41` test period (7,364,160 rows, 1.806 % anomalous), reproduced on
`channel_43` and `channel_45`:

```
count = 0   →  P(anomaly) = 56.7 %   lift 31×   ← 30.7 % of ALL M1 anomalies
count = 1   →  P(anomaly) =  1.26 %  lift 0.70×
count = 10  →  P(anomaly) = 96.6 %   lift 53×
```

**Mechanism.** When no telemetry arrives, ZOH forward-fills the previous value:

```
2007-04-14 20:37:00    0.456281    is_anomaly=1
2007-04-14 20:37:30    0.456281    is_anomaly=1
2007-04-14 20:38:00    0.456281    is_anomaly=1     ← perfectly flat, all labelled anomalous
```

A forecaster predicts "next = same as last", is exactly right, gets ~zero error, and **detects
nothing**. This is a precise mechanical explanation for M1 recall = 0.09.

**It also explains the M1/M2 asymmetry:** M2 has only 136 count-zero rows and 0 % anomalies among
them — and our M2 score is already near-SOTA.

**Fix (not yet implemented):** add label-free `count`, `gap_flag`, `time_since_last_sample`
features. Addresses **requirement R8**. ⚠️ Must be re-derived from the **raw pickles** — the
preprocessed CSVs have already lost this information.

**Caveat to state in any write-up:** we excluded `Category == "Communication Gap"`; these are
genuine Anomaly/Rare-Event annotations where telemetry also stopped. Report scores with *and*
without the feature.

---

## 4. Repository / paths

Everything is committed to **`github.com/shreyaskodithyala2/Capstone`, branch `new`**.
The repo was restructured on 2026-08-12 around **the two winning models**:

```
README.md                    repo map, TL;DR results, quick start, honest status
CLAUDE.md                    this file
docs/  01-DATASET  02-PREPROCESSING  03-METHOD  04-RESULTS
       05-EVALUATION  06-ROADMAP  07-RELATED-WORK
models/README.md             which model won what, and how they relate
models/gdnv2/                ⭐ the shared network + Mission-2 winner (0.9644)
       README.md  src/(9)  ablations/(3)  results/(7)
models/gdnv4/                ⭐ Mission-1 winner (0.8532) + full 5-aspect scoreboard
       README.md  RESULTS.md  src/(4)  ablations/(1)  results/(2)
federated/  README.md  src/(9 incl. official_score.py, telemanom_prune.py)
preprocessing/  README.md  setup_env.sh  run_preprocessing.sh
                build_official_images.sh  docker_build/  patches/
logs/                        13 run logs — the full iteration audit trail
papers/                      REFERENCES.md + 23 PDFs (incl. SMED_Brancato_PHM2025.pdf)
legacy/phase2-analysis/      superseded Phase-2 exploration
```

⚠️ **`models/gdnv4/src/run_v4.py` imports `model.py` and `data.py` from `models/gdnv2/src/`.**
GDNv4 is a better *detection stack* on GDNv2's trained encoder, not a separate network. Keep both.

**Repo → local folder mapping** (the repo renames things):

| Repo path | Local working folder |
|---|---|
| `models/gdnv2/` | `~/Downloads/Capstone/ESA/GDNv2/` |
| `models/gdnv4/` | `~/Downloads/Capstone/ESA/GDNv3/` ← note the name mismatch |
| `federated/` | `~/Downloads/Capstone/ESA/FederatedGDN/` |

**Local working layout** (author's machine — adapt paths):

```
~/Downloads/Capstone/ESA/
├── CLAUDE.md
├── ESA-ADB/          official kplabs-pl clone (code root; cd here for its scripts)
├── ESA-Mission1/     raw Zenodo data (76 channels, 698 telecommands)
├── ESA-Mission2/     raw Zenodo data (100 channels, 123 telecommands)
├── FederatedGDN/     phase-A work (src/, logs/, data_light/, official_score.py replica)
├── GDNv2/            ⭐ the shared network + M2 winner (src/, data/, results/, logs/)
└── GDNv3/            ⭐ GDNv4 detection stack + 5-aspect metrics (src/, results/, RESULTS.md)
```

⚠️ `src/common.py` and `src/official_score.py` contain **absolute paths**. Edit `PREP` and the
mission `raw` paths first.

Activate: `source $(conda info --base)/etc/profile.d/conda.sh && conda activate timeeval`

---

## 5. Environment — the landmines (all already solved)

| Problem | Cause | Fix |
|---|---|---|
| `No matching distribution for requests====2.31.0` | typo in ESA-ADB `setup.py` (four `=`) | `pip install -e . --no-deps --no-build-isolation` |
| `ModuleNotFoundError: natsort` | needed by `infer_anomaly_types.py`, undeclared | install explicitly |
| numpy silently → 2.x | installing torch pulls it, breaking pandas/scipy ABI | `pip install --no-deps torch` + `typing_extensions sympy networkx filelock mpmath fsspec`; **numpy must stay 1.21.6** |
| `request() got an unexpected keyword 'chunked'` | docker SDK 4.4 vs urllib3 ≥2 | `pip install -U docker==7.1.0` |
| Docker image pull `access forbidden` | HPI registry is private | build locally: `preprocessing/build_official_images.sh` |
| `apt-get update` exit 100 | Debian buster EOL | patched base in `preprocessing/docker_build/` |
| `exec /entrypoint.sh: permission denied` | `COPY` drops the exec bit | `chmod +x` in the patched Dockerfile |
| Container start HTTP 500 | TimeEval requests a GPU for every container; macOS has none | `patches/timeeval-docker-nogpu.patch` (`TIMEEVAL_USE_GPU=1` restores) |
| Container exit 137 (OOM) | full 175-col CSVs ≈10 GB, Docker VM has 7.7 GB | `src/make_light_csvs.py` — result-identical trimming |
| amd64 emulation too slow / killed | Rosetta on arm64 | build images **natively** |

Verified stack: Python 3.9, numpy **1.21.6**, pandas 1.5.3, torch 2.8.0 with **MPS working**.

---

## 6. Dataset facts (measured)

- **Zenodo:** <https://zenodo.org/records/15237121> · repo: `github.com/kplabs-pl/ESA-ADB`
- **M1**: 76 channels, 698 telecommands (priority 0:345, 1:323, 2:19, **3:11**), 200 events
  (118 anomalies / 78 rare events / 4 gaps), 14 years, 30 s grid
- **M2**: 100 channels, 123 telecommands (**4** at priority 3), 644 events (31 anomalies / 613 rare
  events), 3.5 years, 18 s grid
- **Lightweight subsets**: M1 = `channel_41…46` (**6**, not 7); M2 = `channel_18…28` (**11**)
- **Preprocessed**: M1 = 175 cols, M2 = 209 cols; `timestamp | channels | telecommands |
  is_anomaly_*`. Labels `0` nominal `1` anomaly `2` rare event `3` gap
- **Correlation**: M1 lightweight **0.97–1.00** (all same subsystem/unit/group); M2 lightweight
  **median 0.03**. This heterogeneity is what makes federation genuinely non-IID
- **Contamination**: M1 lightweight train 1.29 %, test 1.81 %
- Only **priority ≥3** telecommands are kept by the official script
- **Mission 3 is excluded** by the benchmark authors — do not use it

---

## 7. Model design

**GDN** (Deng & Hooi 2021): sensor embeddings → graph structure learning (cosine + top-k) →
graph-attention forecasting → deviation scoring (robust median/IQR normalisation, **max** over
sensors).

**Federated split (FedPer-style)** — M1 has 6 channels, M2 has 11, so weights cannot be averaged
directly:

| Group | Federated? |
|---|---|
| Sensor embeddings (`6×64` vs `11×64`) | ❌ **local / personalized** |
| Shared operators: window transform, attention vector, per-node output head | ✅ **federated** |

The output head had to be redesigned to `Linear(d → 1)` **per node** — the original
`Linear(N·d → N)` scaled with channel count and could not be aggregated.

Loop: local train (FedProx μ=0.01) → upload shared operators only → trimmed-mean aggregate →
broadcast → **per-mission fine-tune** (personalization — critical; without it federation *hurts*
M1).

### Hyperparameters — and two known problems

| Param | Value | Note |
|---|---|---|
| `d` | 64 | untuned; 6 nodes in 64-D is arguably oversized |
| `window` | 100 | ⚠️ Telemanom uses **250** |
| horizon | 1 step | ⚠️ Telemanom predicts **10** |
| epochs | 16 | ⚠️ Telemanom trains up to **1000** + early stopping |
| `topk` | **N−1** | ⚠️ **complete graph → GDN's graph learning is effectively disabled.** Sweep k |

---

## 8. Evaluation — critical

Official metric: `ESAScores(betas=0.5, select_labels={"Category": ["Rare Event", "Anomaly"]})`.

```
Pre = TPe/(TPe+FPe) × (1 − FPt/Nt)      ← penalises false-positive DURATION
Fβe = (1+β²)·Pre·Rece / (β²·Pre + Rece),  β = 0.5   ← precision weighted 4×
```

**`src/official_score.py` is a byte-faithful replica of the TimeEval harness** — validated by
re-scoring the official container output and reproducing `0.9487 / 0.9509 / 0.94` exactly. Use it
to score any new model comparably **without Docker**.

The recipe it replicates (from `timeeval/core/experiments.py`): read `labels.csv`, keep events
**fully inside** the test range, merge the last four `anomaly_types` columns by ID, filter to
target channels **without deduplicating**, then call `metric.score(y_true.drop(columns=["Channel"]),
test_data_scores)`.

⚠️ **Do not compare oracle-threshold numbers to published baselines.** Only val-tuned or
self-thresholding numbers are comparable.

---

## 9. Related work — what matters

- **Atefrad & Karami (2025)** — the only other FL-on-ESA-ADB paper, **same M2 subset**, F0.5 0.86.
  ⚠️ Uses **point/window-wise** metrics (no FP-duration penalty), oversampling, and *stratified*
  train-test splitting (leakage risk on time series), and federates by **time-slicing one
  mission**. Our 0.936 is on a strictly harsher metric with genuinely different spacecraft.
  ✅ Their fine-tuning gain (0.74 → 0.86) independently validates our personalization design.
- **STGLR** (Sensors 2025, doi:10.3390/s25020310) — best available GDN upgrade: two embedding
  tables + trainable projections + antisymmetric combination → **directed** edges; GraphSAGE;
  **GRU + attention** (our GDN has no temporal recurrence); VAE head.
  ⚠️ Their F1 0.97 uses **point adjustment** on SMAP/MSL — not comparable. ⚠️ Their graph is
  **frozen at inference**, same as GDN — a truly per-window dynamic graph is **unclaimed**.
- **Telemanom** — non-pruned 0.178 vs pruned 0.786, *same network*. Post-processing > architecture
  on this benchmark.
- ⛔ **DFSat is RETRACTED** (IEEE notice in the PDF) — must not be cited.
- OPS-SAT checked as a third federated client: **univariate fragments, incompatible.**

Full annotated bibliography: `papers/REFERENCES.md`.

---

## 10. Next steps, in priority order (post-GDNv4)

1. **Seed stability (~1 h, MANDATORY before paper):** rerun GDNv2 M1+M2 and GDNv4 M1 with
   seeds {0,1,2}; report mean±std. M2's margin (+0.015) ≈ 1–2 events.
2. **ADTQC fix (~1 h):** emit the *first index* of each merged run instead of the whole closed
   block. `Nb_Before=23` should collapse at no F0.5 cost. Only aspect where v4 regressed
   against our own v3 (0.186 vs 0.562).
3. **Federated GDNv4 (~half day) — the capstone's title claim.** Embeddings local;
   TCN/attention/head shared; FedProx μ=0.01 + trimmed-mean + personalization fine-tune,
   all already validated on the earlier model. This is a rerun on the winning encoder.
4. **Killer ablation (~half day):** Telemanom-style LSTM **with** the availability features.
   Either outcome strengthens the paper (feature vs model attribution) and pre-empts the
   "baselines didn't have those inputs" critique.
5. **Full-harness confirmation:** wrap GDNv4 as a TimeEval algorithm; produce headline numbers
   through the actual Docker pipeline instead of the validated replica.
6. **Chase SMED on M2** (their 0.971 vs our 0.9644): finish the M2 v4 grid, then sweep top-k
   (we currently run a complete graph, which disables graph learning exactly where M2 needs
   it — median inter-channel correlation 0.03). A MASS "distance to best recent match"
   fusion branch is the other unclaimed idea.
7. Then: STGLR directed-graph ablation, per-window dynamic graph (unclaimed territory),
   full 76/100-channel sets (Kaggle GPU).

**Research framing for the paper** (three contributions, descending confidence):
**C1** availability-aware detection (R8) — proven, and it is a finding about *the benchmark*;
**C2** genuine cross-mission federation (two real spacecraft, not time-slices) with mandatory
personalization; **C3** *personalised detection policies* under federation — the same encoder
needs opposite decision rules per mission (prune: M1 0.8532 / M2 0.014; absolute-z: M1 0.137 /
M2 0.9644), which is worth ~0.8 F0.5 and is not what personalised-FL literature personalises.

Done since the old list: availability features ✅ (R8, the decisive ingredient),
telecommand nodes ✅ (R6), window ≥250 ✅ (RF=253 TCN), EWMA smoothing ✅.
⚠️ Still true: **never feed `is_anomaly` as a model input** (label leakage; train-split labels
as supervision only = R4).

**Compute:** GDNv2 runs comfortably on an M4 Air (MPS, ~40 s/epoch, error passes ~53 s,
~2.7 GB RAM). Full channel sets and TF-GPU baselines → Kaggle free GPU.

---

## 11. Working preferences (from the author)

- **Concise, direct communication.** No filler. Terse code comments.
- **Validate against primary sources** — read the actual script/manifest/paper before asserting.
  Explicitly flag anything inferred-but-unverified.
- **Reproducibility and academic rigour matter** — match published numbers before claiming
  improvement; be honest about what is novel vs. not.
- **When a claim turns out wrong, say so plainly and correct it.** (Two real corrections happened
  in this project: my "ZOH loses the spike between grid points" explanation was mechanically wrong
  — it's *last-sample-wins within a bucket*; and I initially called the `count` feature "modest"
  because my own analysis filtered out the zero-count buckets.)
- Explain pipelines **in the order code executes**, using real variable names.
- The author is comfortable with deep technical detail and prefers it to simplified summaries.

---

## 12. Things NOT to change

- **Do not use Mission 3** — excluded by the benchmark authors; breaks comparability.
- **Do not `pip install timeeval` from PyPI** — the repo's fork has ESA-specific metrics. Install
  editable from the local clone.
- **Do not naively average GDN weights across missions** — different embedding table shapes. Only
  shared operators are federated.
- **Do not feed `is_anomaly` as a model input.** Label leakage.
- **Do not report oracle-threshold scores** alongside published baselines.
- **Do not adopt the Phase-2 90 s grid, `.mean()` resampling, or linear interpolation** — all three
  are actively harmful for anomaly detection (linear interpolation fabricates data across gaps up
  to 13 days).
- **Do not cite DFSat** (retracted), and **do not quote STGLR's 0.97** as comparable.
- **Do not run `mission1_experiments.py` wholesale** — 120-hour timeouts across many algorithms.
- **Do not remove Telemanom pruning from the M1 stack** — the v3 ablation proved it is the
  precision engine (0.999 → 0.444 without it). Conversely, **do not add pruning to M2**
  (0.9644 → 0.014).
- **Do not emit one global score broadcast to all channels** — channel-aware F0.5 drops
  0.474 → 0.123. Aspect 2b requires per-channel predictions.
- **Do not delete `models/gdnv2/`** thinking gdnv4 supersedes it — v4 imports v2's `model.py`
  and `data.py`, and v2 still holds the Mission-2 winner.
- **Do not inherit SMED's "best ESA = 0.842" figure for M2** — they compared only against
  Telemanom-ESA; the real published best is Windowed iForest at 0.949.
- **Do not report only aspect 1a.** Reporting the headline alone hides the ADTQC regression.
- **Do not commit datasets** to the repo (~8 GB raw, ~34 GB preprocessed).
