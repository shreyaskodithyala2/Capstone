# Annotated bibliography

Every paper here was read and assessed for **what it contributes to this specific project**.
Papers judged irrelevant are listed at the bottom with the reason, so nobody re-reads them.

---

## `benchmark/` — the dataset and its evaluation

### ESA-ADB (the benchmark we are measured against) ⭐⭐⭐
`ESA-ADB_Kotowski_NeurIPS2025.pdf`, `ESA-ADB_Kotowski_benchmark.pdf`

Kotowski, Haskamp, Andrzejewski, Ruszczak, Nalepa, Lakey, Collins, Martínez-Heras, De Canio.
*European Space Agency Dataset and Benchmark for Real-World Anomaly Detection in Spacecraft Time
Series.* NeurIPS 2025 Datasets & Benchmarks. Earlier version: arXiv:2406.17826.

**Why it matters:** defines the dataset, the corrected event-wise F0.5 metric, and the baseline
table we compare against. **Cite as the primary dataset reference.**

**Key facts to use:**
- 224 channels, 821 telecommands, 1,430 annotated events (157 anomalies) across 3 missions
- Telecommands grouped by mission impact, 0 (low) → 3 (high)
- **No algorithm satisfies all nine functional requirements.** Worst gaps: **R4** (learn from
  known anomalies), **R6** (auxiliary variables / telecommands), **R7** (test-time learning),
  **R8** (irregular time series), **R9** (memory/timeout at scale)
- Transformers (TranAD, Anomaly Transformer) were **tried and abandoned** by the authors
- Mission 3 is **excluded** from benchmarking
- SMAP/MSL *"should not be used for benchmarking due to unrealistic anomaly density, many trivial
  anomalies, mislabelled ground truth"* — important when reading other papers

### SMED — our closest competitor on ESA-ADB ⭐⭐⭐
`SMED_Brancato_PHM2025.pdf` — Brancato et al. *PHM Society European Conference*, 2025.

**Legitimate**: real ESA-ADB, real ESA metrics, lightweight subsets. Must be cited and beaten
honestly.

| Aspect (Mission 1) | SMED | Ours (GDNv4) |
|---|---|---|
| Event-wise F0.5 | 0.801 | **0.8532** ✅ |
| Channel-aware F0.5 | 0.444 | **0.4740** ✅ |
| Alarming precision | 0.659 | **0.7955** ✅ |
| ADTQC / after-rate | 0.160 / 0.103 | **0.1864 / 0.3429** ✅ |
| **Mission 2 event-wise F0.5** | **0.971** ✅ | 0.9644 |

⚠️ **Their comparison column is misleading.** SMED benchmarks only against Telemanom-ESA, so its
"best ESA" for Mission 2 reads **0.842** — the actual published best is Windowed iForest at
**0.949**. Do not inherit that error when citing them.

### OPS-SAT benchmark
`OPSSAT-AD_Ruszczak_2025.pdf` — Ruszczak, Kotowski, Evans, Nalepa. *Scientific Data*, 2025.

Another real satellite telemetry AD dataset from the same group.

⚠️ **Checked as a potential third federated client — it does not fit.** It contains 2,123 short
**univariate** fragments across 9 channels. Our GDN needs continuous multivariate streams to build
a sensor graph. **Structurally incompatible — do not pursue.**

---

## `models/` — architectures

### GDN — our base model ⭐⭐⭐
`GDN_Deng-Hooi_AAAI2021.pdf` — Deng & Hooi. *Graph Neural Network-Based Anomaly Detection in
Multivariate Time Series.* AAAI 2021. arXiv:2106.06947.

Four components: sensor embedding → graph structure learning (cosine + top-k) → graph-attention
forecasting → graph deviation scoring. Evaluated on SWaT and WADI.

**Use:** the architecture we implement. See `docs/03-METHOD.md`.

### STGLR — the best available upgrade ⭐⭐⭐
`STGLR_Lai_Sensors2025.pdf` — Lai, Zhu, Li, Lan, Zuo. *STGLR: A Spacecraft Anomaly Detection
Method Based on Spatio-Temporal Graph Learning.* Sensors 2025, 25(2), 310. doi:10.3390/s25020310

Explicitly critiques GDN. Three modules: dynamic graph learning (two embedding tables + trainable
projections + antisymmetric combination → **directed** edges), GraphSAGE + attention-GRU, and a
VAE reconstruction head.

**Take:** the directed graph construction and the GRU (our GDN has **no temporal recurrence**).

⚠️ **Two cautions.** (1) Their F1 = 0.97 uses **point adjustment** on SMAP/MSL — heavily inflated
and not comparable to corrected event-wise F0.5. (2) Despite the name, their graph is **frozen at
inference**, same as GDN; "dynamic" means updated *during training*. A genuinely per-window graph
remains **unclaimed**.

### Telemanom — the score to beat ⭐⭐⭐
`Telemanom_Hundman_KDD2018.pdf` — Hundman et al. *Detecting Spacecraft Anomalies Using LSTMs and
Nonparametric Dynamic Thresholding.* KDD 2018. arXiv:1802.04431.

LSTM forecaster + nonparametric dynamic thresholding + **pruning**. The ESA-ADB variant is the
strongest baseline on Mission 1 (**0.786**).

**The single most important lesson in this bibliography:** non-pruned scores **0.178**, pruned
scores **0.786** — *same network*. On this benchmark, post-processing outweighs architecture.
We ported this in `federated/src/telemanom_prune.py`; it lifted our M1 from 0.080 → 0.337 with precision 1.00.

### DeepAnT
`DeepAnT_Munir_2019.pdf` — CNN-based unsupervised forecasting for TS anomaly detection.
**Use:** related-work baseline; an alternative forecaster if GDN plateaus.

### DAGMM
`DAGMM_Zong_ICLR2018.pdf` — Deep autoencoding Gaussian mixture model.
**Use:** standard reconstruction-family baseline for related work.

### Probabilistic GNN for MTS anomaly detection
`ProbabilisticGNN_MTS-AD.pdf` — arXiv:2111.08082.
**Use:** related work; probabilistic alternative to GDN's deterministic graph.

---

## `federated/` — the FL side

### Atefrad & Karami — the closest prior work ⭐⭐⭐
`Atefrad-Karami_FL-on-ESA-ADB_2025.pdf` — Univ. of East London, 2025.

**The only other FL-on-ESA-ADB paper, and it uses the identical Mission 2 lightweight subset
(channels 18–28).** Reports F0.5 = 0.86, precision 0.93, AUC 0.85 with LSTM + FedAvg +
fine-tuning; 1.8 MB per round; scales to 10 clients.

**Not comparable to our numbers, and the differences favour us:**

| | Them | Us |
|---|---|---|
| Metric | *"standard binary classification metrics"* — point/window-wise | official corrected event-wise F0.5 |
| Federation | clients = **time blocks of one mission** | **two different spacecraft** |
| Imbalance handling | oversampling + **stratified** train-test split | none; chronological split preserved |
| Score | 0.86 | **0.936** |

⚠️ Their *"stratification during the train-test split"* implies shuffling — **leakage risk on time
series**.

✅ **They independently validate our design:** their fine-tuning lifted F0.5 from 0.74 → 0.86 —
the same personalization effect we found.

**Cite as the primary related work and differentiate on these three axes.**

### FedAvg — foundational
`FedAvg_McMahan_AISTATS2017.pdf` — McMahan et al. arXiv:1602.05629. The baseline FL algorithm.

### Advances and Open Problems in Federated Learning
`FL-Advances-and-Open-Problems.pdf` — Kairouz, McMahan et al. arXiv:1912.04977.
**Use:** authoritative source for framing non-IID challenges and personalization.

### On-board FL for satellite clusters with inter-satellite links
`OnBoard-FL-Satellite-ISL_Razmi.pdf` — arXiv:2307.08346.
**Use:** realistic satellite FL topology; supports the bandwidth/privacy motivation.

### FedHC — hierarchical clustered FL for satellite networks
`FedHC_Satellite-Networks_Liu.pdf` — arXiv:2502.12783.
**Use:** relevant **only if** scaling beyond 2 clients.

### Hierarchical FL across heterogeneous cellular networks
`Hierarchical-FL_Abad.pdf` — arXiv:1909.02362. Multi-tier aggregation.

### FL for predictive maintenance and quality inspection
`FL-Predictive-Maintenance_Pruckovskaja.pdf` — arXiv:2304.11101. Industrial FL comparison.

---

## `related/` — satellite anomaly detection context

### A Comparison of Deep Learning Architectures for Spacecraft Anomaly Detection
`DL-Architectures-Comparison_Lakey.pdf` — Lakey & Schlippe. arXiv:2403.12864.
**Note: Lakey is an ESA-ADB co-author.** Useful architecture-comparison context.

### Data-driven health monitoring (Yairi et al.)
`Health-Monitoring_Yairi.pdf` — probabilistic clustering + dimensionality reduction on satellite
housekeeping data (JAXA). Classical baseline, pre-deep-learning.

### Unsupervised ML for spacecraft anomaly detection in WebTCAD
`WebTCAD_Polson_2019.pdf` — LASP. Practical operational survey.

### Kaggle top-3 hierarchical ensemble
`Kaggle-Hierarchical-Ensemble.pdf` — arXiv:2605.06681. Shapelet + statistical features → XGBoost →
logistic-regression cross-channel aggregator. 1st public leaderboard (0.931), 3rd private (0.853).
**Use:** evidence that ensembles and hand-crafted features work well on this data.

---

## `phase2/` — our own prior work

`122_Phase2_ESA_Report.pdf`, `122_Phase2_ESA_PPT.pdf` — the graded Phase 2 submission (channel
interval analysis, ZOH/aggregate classification, custom preprocessing, initial GDN).

Assessed in `docs/02-PREPROCESSING.md`. Summary: the interval analysis is correct and valuable
(their "~79 readings per 30 s burst" — we measured 81); the **`count` feature is the best idea in
it**; the 90 s grid, `.mean()` resampling, and linear interpolation should **not** be adopted.

⚠️ §4.1's claim that ESA-ADB has *"no published work addressed with GNN methods"* is **outdated** —
arXiv 2607.07335 (July 2026) benchmarks GCN and GAT. Narrow the novelty claim to **GDN
specifically** plus **cross-mission federation**.

---

## Excluded, with reasons

Kept out deliberately so nobody wastes time on them.

| Paper | Why excluded |
|---|---|
| **DFSat** (Moustafa et al.) | ⛔ **RETRACTED.** The PDF's first page is an IEEE *"Notice to Reader"* stating it *"should not be considered for citation purposes"* (authorship dispute). **Remove from any reference list.** |
| Deep-FL threat detection (Salim et al.) | Cybersecurity intrusion detection, not telemetry anomaly detection — different task |
| Forecasting solar flares with a transformer | Space weather, unrelated method and target |
| Impact of solar irradiance on LEO orbital decay | Orbital mechanics, no ML methodology to import |
| Satellite anomalies and their causes | Space-environment survey; motivation only |
| CubeSats & Nanosatellites 2024 statistics | Market statistics; motivation only |
| Satellite swarm survey / conceptual design | Mission design, not anomaly detection |
| Autonomous control of satellite swarms (vision-based) | Formation control, unrelated |
| NASA State-of-the-Art Small Spacecraft Technology | Hardware reference |
| University CubeSat project management | Programme management |
| Agent-based artificial immune system for fault detection | Different paradigm; no transferable component |
| Autoencoder-LSTM edge accelerator (FPGA) | Hardware implementation; relevant only for an on-board deployability claim |

---

## Reading order for someone new

1. `benchmark/ESA-ADB_Kotowski_NeurIPS2025.pdf` — what the data and metric are
2. `models/GDN_Deng-Hooi_AAAI2021.pdf` — our base model
3. `models/Telemanom_Hundman_KDD2018.pdf` — the score to beat, and why pruning matters
4. `federated/Atefrad-Karami_FL-on-ESA-ADB_2025.pdf` — the closest competitor
5. `models/STGLR_Lai_Sensors2025.pdf` — where to improve GDN next
