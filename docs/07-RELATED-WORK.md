# 7 · Related work — what exists, and where we sit

Full annotated bibliography with all 22 PDFs: [`../papers/REFERENCES.md`](../papers/REFERENCES.md).
This page is the short version: who we compete with, and what is safe to cite.

---

## The benchmark itself

**ESA-ADB** — Kotowski et al., *"European Space Agency Benchmark for Anomaly Detection in Satellite
Telemetry"*, KP Labs / Airbus DS / ESA.
Repo: <https://github.com/kplabs-pl/ESA-ADB> · Data: <https://zenodo.org/records/15237121>

Defines the missions, the preprocessing, the 5-aspect metric hierarchy, and a list of **unmet
requirements** the authors say no algorithm satisfies. Two of those are ours:

| Req | What it asks | Our status |
|---|---|---|
| **R6** | Use telecommands as model input | ✅ input-only exogenous graph nodes |
| **R8** | Handle irregular / gappy time series | ✅ availability features — the decisive ingredient |
| R4 | Learn from known anomalies | ✅ train labels used for threshold selection only |

---

## Direct competitors on ESA-ADB

### Telemanom-ESA (published baseline, ESA's own best on M1)

LSTM forecaster + dynamic thresholding + **pruning**. Published: **0.786** M1 / 0.842 M2 pruned;
**0.178** M1 non-pruned — *same network*.

**The lesson we took from it:** on this benchmark, post-processing beats architecture. A 0.178 →
0.786 jump from pruning alone is larger than any architectural gain in the literature. We ported
`find_epsilon` and `prune_anoms` faithfully (`federated/src/telemanom_prune.py`) and pruning is the
precision engine in GDNv4.

### Windowed iForest (published best on M2)

**0.949** M2. Reproduced exactly through the real Docker harness: **0.9487**. This is our
validation anchor.

### SMED — Brancato et al., PHM Society 2025 ⚠️ our closest competitor

| | SMED | Ours |
|---|---|---|
| Mission 1 EW F0.5 | 0.801 | **0.8532** ✅ |
| Mission 2 EW F0.5 | **0.971** ✅ | 0.9644 |
| M1 channel-aware | 0.444 | 0.4740 ✅ |
| M1 alarming precision | 0.659 | 0.7955 ✅ |
| M1 ADTQC | 0.160 / after-rate 0.103 | 0.1864 / 0.3429 ✅ |

**Legitimate** — real ESA-ADB, real metrics. Must be cited and must be beaten honestly.

⚠️ Read their comparison column carefully: SMED benchmarks only against Telemanom-ESA, so their
"best ESA" for M2 reads **0.842** when the actual published best is Windowed iForest at **0.949**.
Do not inherit that error.

### DC-VAE-ESA

Weak on aspect 1a (0.003 M1) but the **ADTQC leader (0.901)**. Concrete proof of the
precision/timing trade-off documented in [`05-EVALUATION.md`](05-EVALUATION.md).

---

## Federated learning on satellite telemetry

### Atefrad & Karami (2025) — the only other FL-on-ESA-ADB paper

Same M2 lightweight subset. Reports F0.5 **0.86**.

⚠️ **Not directly comparable**, for three reasons:
1. **Point/window-wise metrics** — no false-positive *duration* penalty, the single harshest part
   of ESA's corrected event-wise score.
2. **Oversampling** and **stratified** train/test splitting — leakage risk on time series.
3. Federates by **time-slicing one mission**, not across spacecraft.

Our 0.936 (and now 0.9644 solo) is on a strictly harsher metric with genuinely different
spacecraft.

✅ **But it validates our design:** their fine-tuning gain (0.74 → 0.86) independently reproduces
the personalisation effect we measured (federation without fine-tuning *hurts* M1).

### ⛔ DFSat — **RETRACTED**

The PDF carries an IEEE retraction notice. **Do not cite it.** Listed here only so nobody
re-discovers it and uses it.

---

## Architecture literature we drew on

### GDN — Deng & Hooi, AAAI 2021

Our base. Sensor embeddings → cosine top-k graph → graph-attention forecasting → deviation
scoring. We changed: availability features (new), TCN encoder (new), telecommand nodes (new),
per-node head (needed for federation).

### STGLR — *Sensors* 2025, doi:10.3390/s25020310

The best available GDN upgrade. Two embedding tables + trainable projections + antisymmetric
combination → **directed** edges; GraphSAGE aggregation; **GRU + attention** temporal path; VAE
head.

⚠️ Their F1 = 0.97 uses **point adjustment** on SMAP/MSL. **Not comparable to anything here** —
do not quote it.
⚠️ Their graph is **frozen at inference**, exactly like GDN. (An earlier note in this project
claimed it was dynamic per window; that was wrong and is corrected here.) A truly per-window
dynamic graph therefore remains **unclaimed territory** — see [`06-ROADMAP.md`](06-ROADMAP.md).

Adoptable from STGLR: directed edges via two embedding tables (cheap ablation, worth running).

### Telemanom — Hundman et al., KDD 2018

Origin of dynamic thresholding and pruning. Uses window 250, horizon 10, up to 1000 epochs. Our
TCN's RF = 253 was chosen to match its temporal context.

---

## Datasets we evaluated and rejected

| Dataset | Verdict |
|---|---|
| **ESA Mission 3** | ⛔ Excluded by the benchmark authors. Using it breaks comparability. |
| **OPS-SAT** | ❌ Univariate fragments — incompatible as a third federated client. |
| SMAP / MSL | ⚠️ Widely used but heavily criticised (trivially solvable, point-adjustment inflation). Fine as an extra federated client; useless as a headline benchmark. |

---

## Where we actually stand

**Ahead:** Mission-1 event-wise F0.5 (0.8532 — best we can find anywhere), Mission-2 affiliation
F0.5, and the R6 + R8 requirements.

**Behind:** Mission-2 event-wise F0.5 (SMED 0.971), ADTQC, channel-aware F0.5, and recall.

**Unclaimed:** genuine cross-mission federation on ESA-ADB, per-window dynamic graphs, and
personalised *detection policies* under federation.
