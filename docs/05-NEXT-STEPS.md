# 5 · Where we are, and what to do next

---

## Current state, in one paragraph

Preprocessing is **done and verified** for both missions (it reproduces a published benchmark
number exactly). A cross-mission federated GDN **runs end to end on a laptop** and shows
federation with personalization improving both missions over solo training. Mission 2 is
competitive with the benchmark's best (0.936 vs 0.949). Mission 1 is not (0.337 vs 0.786) — but we
have identified the mechanism, and it is fixable.

---

## The four contributions available for a paper

| # | Contribution | Addresses | Status |
|---|---|---|---|
| **1** | **Telemetry-availability-aware detection** — 30.7 % of M1 anomalies are erased by standard ZOH preprocessing; restoring them via count / gap / time-since features | **R8** (irregular time series) | ✅ **Measured & verified, unreported elsewhere** |
| **2** | **Telecommand-conditioned graph** — commands as directed exogenous nodes | **R6** (auxiliary variables) | Buildable now; data already preprocessed |
| **3** | **Directed + temporal GDN** — STGLR's antisymmetric graph construction + a GRU | Model capacity | Buildable now |
| **4** | **Cross-mission federation with personalization** | Novel setting | ✅ Working (0.936 on M2) |

*(R6 and R8 are ESA-ADB functional requirements that **no benchmark algorithm satisfies**.)*

Contribution **#1 is the strongest** — it's an empirical finding about the benchmark itself,
verified across five channels on both missions, not just a model tweak.

---

## Recommended order

### ① Data-availability features — biggest measured effect, ~1 hour

**Why first:** we have direct evidence that 30.7 % of Mission 1's anomalies are invisible to any
forecasting model under ZOH. Nothing else has this much measured upside.

Add three label-free features (computable from raw telemetry at inference time):

```
count       : raw samples in the window
gap_flag    : 1 if count == 0
time_since  : seconds since the last real sample   ← probably the strongest
```

Feed them as extra per-node input channels. Report M1 **with and without**, and state the caveat
about telemetry loss vs. anomaly explicitly (see `docs/02-PREPROCESSING.md`).

**Note:** this requires re-deriving `count` from the **raw** pickles — the preprocessed CSVs have
already lost it. A small script that emits a per-grid-point count column alongside the existing
CSVs is enough.

### ② Telecommand-conditioned graph — biggest novelty, half a day

Our GDN currently loads **only channel columns** — verified in `common.py`:

```python
df = pd.read_csv(csv, usecols=chs)   # ← channels ONLY; the 11 telecommand columns unused
```

**Recommended approach — exogenous graph nodes:**

```
nodes: [ch41…ch46, tc36, tc38, …, tc376]      (6 + 11 = 17)

edge mask:
   channel → channel     : learned
   telecommand → channel : learned      ← NEW
   channel → telecommand : FORCED 0     (a sensor cannot cause a command)
   telecommand → telecommand : FORCED 0
```

Only the 6 channels are forecast targets; telecommands are inputs only.

Three reasons this is the best option: it fits the graph formalism naturally; the learned
`tc → channel` edges become an **interpretable operational artifact** (which commands affect which
sensors — genuinely useful, and a great figure, since ESA anonymised the names specifically to
prevent lookup); and the causal direction constraint pairs perfectly with STGLR's antisymmetric
construction.

Simpler alternatives: concatenate the 11 columns to every node's input (5 lines, no
interpretability), or use an exponential decay window after each firing (τ ≈ 10–30 min) to model
lingering effects.

> ⚠️ **Do not feed `is_anomaly` as a model input.** That is label leakage and invalidates every
> result. The correct uses are: supervision during **training only** on the **train split** (this
> addresses unmet requirement **R4**), evaluation, and post-hoc correlation analysis. A real
> satellite knows what commands it received; it does not know whether the current moment is
> anomalous.
>
> The behaviour you want emerges naturally from forecasting: with telecommands as input, a
> commanded jump becomes *predictable* → low error → no alarm; the same jump with no command → high
> error → alarm.

### ③ Close the Telemanom configuration gap — cheap, likely significant

Our model is smaller and trained far less than the baseline it's compared against:

| | Telemanom | Ours |
|---|---|---|
| History window | **250** (2 h) | 100 (50 min) |
| Predicts ahead | **10** steps | 1 step |
| Training | up to 1000 epochs + early stopping | **16 epochs** |
| Error smoothing | EWMA + `error_buffer=100` | moving average, 10 |

Match these before concluding anything about architecture. Also run the **top-k sweep** (see
`docs/03-GDN-MODEL.md`) — GDN's graph learning is currently disabled.

### ④ Adopt STGLR's directed graph + GRU

From `papers/models/STGLR_Lai_Sensors2025.pdf`:

- **Two embedding tables** (source role + target role) with **trainable projections**, combined
  antisymmetrically so exactly one direction survives per pair → real directed edges
- **GRU + attention** over the window — our GDN has **no temporal recurrence at all**

⚠️ **Do not cite their F1 = 0.97 as a target.** It uses **point adjustment** on SMAP/MSL, datasets
the ESA-ADB authors say *"should not be used for benchmarking"*. Take the architecture, not the
number.

### ⑤ The genuinely novel gap — per-window dynamic graphs

**Neither GDN nor STGLR recomputes the graph at inference.** STGLR's "dynamic" means *updated
during training*; after training it is frozen, exactly like GDN.

So nobody has built a graph recomputed **per test window** that uses the **change in graph
structure itself** as the anomaly score. On our data that is precisely the signal:

```
19:26:00  ch41 0.803  ch42 0.793  ch43 0.774  …   ← normal: tightly coupled
19:26:30  ch41 0.970  ch42 0.933  ch43 0.000  …   ← ch43 DECOUPLES
```

A static graph is identical in both rows. A per-window graph would show channel 43's edges
collapsing. **The graph delta is the detection signal** — a new scoring mechanism, not a new block.

This is especially meaningful in the **federated cross-mission** setting, because the two
spacecraft have structurally different graphs (M1 dense ~1.0 correlation, M2 sparse ~0.03,
measured). Federate the graph-*learning operators* while each mission keeps its own topology.
Nobody has done federated graph-structure learning across heterogeneous spacecraft.

---

## Compute guidance

| Phase | Where | Why |
|---|---|---|
| Lightweight subsets (all current work) | **M4 MacBook Air, PyTorch MPS** | ~600 k windows, ~23 s/epoch, ~1 GB RAM. Comfortable |
| Full 76/100-channel sets | **Kaggle notebooks (free, 30 GPU-h/week)** | Best free option |
| Telemanom / DC-VAE baselines | **Kaggle or Colab** | Need `tensorflow:2.5.1-gpu` (amd64 + CUDA) |
| Long federated sweeps | vast.ai / RunPod (~$0.3–0.7/h) | Only if free tiers bottleneck |

Buying an RTX 5090 is **not** worth it for this project.

---

## Publication framing

**Realistic venue:** workshop, student track, IEEE Aerospace, or arXiv — not NeurIPS main track.
That is a good outcome for a capstone.

**Suggested title:**
> *A Telecommand-Aware, Availability-Robust Federated Graph Deviation Network for Cross-Mission
> Spacecraft Anomaly Detection*

**The claim to make:** cross-mission federated GDN achieves near-centralised accuracy **without
sharing raw telemetry between spacecraft**, plus a previously unreported preprocessing artifact
that explains why forecasting models underperform on Mission 1.

**Do not claim** "we beat the benchmark" until Mission 1 is competitive under official
fixed-threshold scoring.

**Correction needed in the Phase-2 report:** §4.1 claims ESA-ADB has *"no published work addressed
with GNN methods"*. That is now outdated — arXiv 2607.07335 (July 2026) benchmarks GCN and GAT.
Narrow the claim to **GDN specifically** plus **cross-mission federation**, both still unclaimed.

---

## Immediate next action

Implement **① data-availability features** and re-run Mission 1. It has the largest measured
upside, takes about an hour, needs no new dependencies, and directly addresses an unmet benchmark
requirement.
