# 3 · Method — the models, step by step

This document walks the full pipeline **in the order the code executes**, using the real variable
and file names. If you read one file, read this one.

```
raw pickles ──► extract_features.py ──► data.py ──► model.py (train.py) ──► errors
                                                                              │
                                            ┌─────────────────────────────────┤
                                            ▼                                 ▼
                                  m2_absolute.py                        run_v4.py
                                  MISSION 2 → 0.9644                MISSION 1 → 0.8532
```

---

## Step 0 · Why a graph model at all

Spacecraft telemetry is **multivariate and causally coupled**: a thermal channel responds to a
heater command, a power channel responds to a mode change. Three families were on the table:

| Family | Verdict |
|---|---|
| Point detectors (iForest, HBOS, PCC) | Reproduced them. **0.00007** on M1. They flag isolated points; the metric punishes FP duration. Dead end. |
| Per-channel forecasters (Telemanom) | The published winner (0.786 pruned). But treats every channel independently — throws away cross-channel structure. |
| **Graph forecasters (GDN)** | Learn *which channels depend on which*, forecast each channel from its neighbours, flag deviation. Fits the physics. **Chosen.** |

GDN (Deng & Hooi, AAAI 2021) is the base. We changed three things — availability features, a TCN
encoder, and telecommand nodes — and those changes are where the gain comes from.

---

## Step 1 · `extract_features.py` — recover what preprocessing destroyed

**Runs on the RAW pickles, not the preprocessed CSVs.** This is non-negotiable: the official CSVs
have already applied Zero-Order-Hold and the availability information is gone.

For every channel, for every point on the official grid:

```python
idx      = np.searchsorted(raw_ns, g_ns, side="right")
prev_idx = np.searchsorted(raw_ns, g_ns - grid_step_ns, side="right")
count    = (idx - prev_idx).astype(np.uint16)      # samples that arrived in this window
tsince   = (g_ns - raw_ns[idx - 1]) / 1e9          # seconds since the last real sample
```

Plus `tcdt` — seconds since the last priority-≥3 telecommand firing.

**Why this is the most important step in the project.** Measured on `channel_41` test
(7,364,160 rows, 1.806% anomalous), reproduced on `channel_43` and `channel_45`:

| `count` | P(anomaly) | Lift | Share of all M1 anomalies |
|---|---|---|---|
| **0** | **56.7%** | **31×** | **30.7%** |
| 1 | 1.26% | 0.70× | — |
| 10 | 96.6% | 53× | — |

When no telemetry arrives, ZOH forward-fills the previous value. The series is perfectly flat, a
forecaster predicts it exactly right, the error is ~zero, and **nothing is detected** — even
though those rows are labelled anomalous. That is a mechanical explanation for the published
Mission-1 recall of 0.09.

It also explains the M1/M2 asymmetry: Mission 2 has only **136** count-zero rows and **0%**
anomalies among them, which is why every algorithm scores higher there.

These features are **label-free** — derived purely from arrival timestamps — so using them is not
leakage. Full derivation: [`02-PREPROCESSING.md`](02-PREPROCESSING.md).

> ⚠️ Honest caveat for any write-up: we exclude `Category == "Communication Gap"` events, but the
> remaining Anomaly/Rare-Event annotations *also* include cases where telemetry stopped. Report
> scores with **and** without the feature.

---

## Step 2 · `data.py` — build the node feature tensor

Every node gets 3 features per timestep, so channels and telecommands share one tensor shape:

| Node type | Feature vector |
|---|---|
| Channel | `[ value , gap_flag , log10(1 + tsince) / 7 ]` |
| Telecommand | `[ exp(−tcdt / 1800) , 0 , 0 ]` |

`TC_TAU = 1800 s` — a command's influence decays with a 30-minute half-life-ish envelope, so the
model sees "a command fired *recently*" rather than a one-hot spike it would smooth away.

Result: `(T, N, 3)` where `N = n_channels + n_telecommands`
(M1: 6 + 11 = 17 nodes; M2: 11 + 4 = 15 nodes).

**`is_anomaly` is never an input.** Train-split labels are used only to select thresholds.

---

## Step 3 · `model.py` — the network (38,049 parameters)

### 3a. Temporal encoder — causal dilated TCN

```
6 × Conv1d, kernel 5, dilations 1, 2, 4, 8, 16, 32
receptive field = 1 + Σ (k−1)·d = 1 + 4·63 = 253 steps ≈ 2.1 h on M1's 30 s grid
→ 1×1 conv projection to d = 64
```

Left-padded only, so no future information leaks.

**Why a TCN and not the GRU the plan originally called for.** Both model temporal context; only
the TCN can encode the *entire* series in one forward pass. Both 7.36M-row error passes finish in
**~53 seconds** on Apple MPS. A per-window GRU re-runs a 250-step recurrence at each of 7.36M
positions — 30–60 minutes per pass. That single decision is what made six full design iterations
possible on a laptop.

It also fixes a documented weakness: the Phase-2 GDN used `window = 100`; Telemanom uses 250.
RF = 253 matches the published setting.

### 3b. Graph construction — `get_adj()`

```python
sim = cosine_similarity(embeddings)         # learned node embeddings, 64-D
A[:n_ch, :n_ch]  = topk(sim, k)             # channel → channel : LEARNED
A[:n_ch, n_ch:]  = 1.0                      # telecommand → channel : ALWAYS ALLOWED
A[n_ch:, :]      = 0 ; A[i, i] = 1          # telecommand rows : self-loop only
```

Two hard-coded asymmetries encode real physics:
- **A command can affect a sensor; a sensor cannot issue a command.** Forbidding
  `channel → telecommand` shrinks the hypothesis space and makes the surviving edges
  interpretable — they recover *which command drives which sensor*, a mapping ESA deliberately
  anonymised. This addresses unmet requirement **R6**.
- Telecommands carry no forecast target, so they never receive messages.

> ⚠️ Known open item: on the lightweight subsets we run `topk = N−1`, i.e. a complete channel
> graph — which effectively disables GDN's graph *learning*. A top-k sweep is on the roadmap. On
> M1 this is defensible (the 6 lightweight channels correlate 0.97–1.00); on M2 (median
> correlation 0.03) it is a real loss and likely leaves points on the table.

### 3c. Graph attention — decomposed

Standard GAT concatenates `[g_i ‖ g_j]` and materialises an `N × N × 2d` tensor. At 7.36M
timesteps that does not fit. We use the algebraically equivalent decomposition:

```
e_ij = LeakyReLU( a₁ · g_i + a₂ · g_j )
```

Two `N × 1` vectors broadcast into `N × N` — same result, a fraction of the memory. Masked by the
adjacency, then softmax over each node's neighbours.

### 3d. Output head — per node, `Linear(d → 1)`

The original GDN head is `Linear(N·d → N)`: its shape depends on the channel count, so weights
from a 6-channel client and an 11-channel client cannot be averaged. A federated smoke test caught
this immediately:

```
RuntimeError: stack expects each tensor to be equal size, but got [384,384] and [704,704]
```

With a shared per-node head, **every** federated operator is dimension-independent.

### 3e. `series_errors()`

One full-series forward pass → `|forecast − actual|` per channel per timestep → `(T, n_ch)`.
This array is the sole interface between the model and both detection stacks.

---

## Step 4 · `train.py`

MSE on 1-step-ahead forecasts, Adam, ~40 s/epoch on MPS, ~2.7 GB RAM. Trained on the **train split
only** (pre-2007 for M1, pre-Oct-2001 for M2). Operating points are chosen afterwards, on the
train period, using train labels.

---

## Step 5a · Mission 2 detection — `m2_absolute.py` → **0.9644**

```
errors → EWMA(span = 15) → robust z (per-channel median / IQR, max over channels) → z > 20
```

That is the whole rule. `z* = 20` was tuned on the train period and **transferred as an absolute
value**, pre-registered before test scoring.

| Variant | Test EW F0.5 |
|---|---|
| **absolute z\* = 20 (reported)** | **0.9644** |
| quantile transfer instead | 0.9260 |
| test-set q0.98 | 0.9798 — *oracle, not reportable* |

---

## Step 5b · Mission 1 detection — `run_v4.py` → **0.8532**

Mission 1 needs a completely different stack. Four branches OR-ed **per channel**:

```
errors
  │
  ├─ ZERO the errors inside count==0 runs, +30-step margin
  │    Gap re-acquisition spikes reach z ≈ 1149 and poison pruning's sorted-peak
  │    logic. Test has 16,676 gap runs vs 3,196 in train — justified entirely from
  │    LABEL-FREE input statistics.
  │
  ├─ A · Telemanom dynamic threshold + prune (p = 0.3)   ← the precision engine
  ├─ B · multi-span block-local robust z                 ← the recall lifter
  │        spans {1,3,15,60,240}, 24 h blocks, max over spans, q = 0.9999
  ├─ C · per-channel gap detector (count==0 run ≥ 10)
  └─ D · telecommand-gated threshold (×0.6 within 2 h of a priority-3 firing)
  │
  ▼
morphological CLOSING (bridge gaps ≤ 60 steps)  →  MIN-RUN filter (drop runs < 3)
  │
  ▼
(T × n_channels) binary predictions
```

**Pre-registered config** (train F0.5 = 0.7748, selected from a 432-config train-only grid, frozen
before the single test evaluation):
`p=0.3 · zmode=local · q=0.9999 · close=60 · minlen=3 · gapL=10 · boost=0.6`

### Every branch has measured justification

| Branch | Evidence |
|---|---|
| **A · pruning** | Ablation `run_v3.py` removes it, keeps everything else: precision **0.999 → 0.444**, F0.5 → **0.3827**. Pruning *is* the precision. |
| **B · block-local z** | `miss_analysis.py` found every missed event sat **below the global background p99.9**. In isolation, an identical-magnitude event scores **z = 26.8 in a quiet stretch vs 3.6 in a noisy one** under local normalisation. |
| **C · gap detector** | Alone, it scores **0.412 at precision 1.000** on M1 test. |
| **D · telecommand gate** | **52.8%** of rare events have a priority-3 command within ±2 h, vs **10.3%** of genuine anomalies, against a **0.4%** base rate. |
| **closing** | GDNv2 fired ~4.4 fragments per real event (alarming precision 0.228). Closing → **0.7955**, no F0.5 cost. |
| **min-run 3** | Removes isolated single-step blips that cost precision for nothing. |

**Why predictions are per-channel and not one global signal:** broadcasting one signal to all
channels gives channel-aware F0.5 = **0.123**. Per-channel emission gives **0.474**. Aspect 2b is
part of the benchmark; a global signal simply cannot compete on it.

---

## Step 6 · Federation — `federated/src/`

Validated on the earlier (pre-GDNv2) model; **not yet re-run on GDNv4**.

Mission 1 has 6 channels, Mission 2 has 11, so weights cannot be averaged wholesale. FedPer-style
split:

| Component | Federated? | Why |
|---|---|---|
| Node embeddings (`6×64` vs `11×64`) | ❌ local | shapes differ; they encode *this* spacecraft |
| TCN encoder, attention vectors, per-node head | ✅ shared | dimension-independent by construction |

Round: local train (FedProx, μ = 0.01) → upload shared operators → **trimmed-mean** aggregate →
broadcast → **per-mission fine-tune**.

Ablation (oracle threshold — **internal comparison only, never report alongside published numbers**):

| | Solo | Fed, no fine-tune | Fed + fine-tune |
|---|---|---|---|
| M1 | 0.1765 | 0.1370 ❌ | **0.1892** ✅ |
| M2 | 0.7739 | 0.7820 | **0.8050** ✅ |

**Personalisation is not optional** — without the fine-tune step, federation actively *hurts*
Mission 1. This independently matches Atefrad & Karami (2025), who report the same pattern
(0.74 → 0.86 with fine-tuning).

And the two detection stacks above sharpen the point: the missions need **opposite detection
policies** (pruning wins M1 at 0.8532 but destroys M2 at 0.014). That argues for personalising the
*decision layer*, not just the weights — which is the roadmap's core research claim.
See [`06-ROADMAP.md`](06-ROADMAP.md).
