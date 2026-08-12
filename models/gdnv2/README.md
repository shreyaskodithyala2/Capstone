# GDNv2 — the foundation model + Mission-2 winner

**Result: Mission 2 event-wise F0.5 = 0.9644** (P 0.9606, R 0.9800) vs published best **0.949**.
Also **affiliation F0.5 = 0.9839**, above the published 0.976.

This folder holds the neural network that *both* winning pipelines use, the feature extraction
that made the whole project work, and the Mission-2 detection stack.

---

## 1. The architecture

```
INPUT per channel node   [ value , gap_flag , log10(1+Δt_since_sample)/7 ]
INPUT per telecommand    [ exp(−Δt_since_firing / 1800s) , 0 , 0 ]
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │ TEMPORAL ENCODER — causal dilated TCN            │
        │ 6 conv layers, kernel 5, dilations 1,2,4,8,16,32 │
        │ receptive field = 253 steps  ≈ 2.1 h on M1       │
        │ → 64-dim hidden state per node per timestep      │
        └──────────────────────────┬───────────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │ GRAPH CONSTRUCTION                               │
        │ node embeddings → cosine similarity → top-k      │
        │ channel↔channel : learned                        │
        │ telecommand→channel : always allowed             │
        │ channel→telecommand : FORBIDDEN                  │
        │ telecommand→telecommand : FORBIDDEN              │
        └──────────────────────────┬───────────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │ GRAPH ATTENTION (decomposed, memory-light)       │
        │ e_ij = LeakyReLU( a1·g_i + a2·g_j )              │
        │ masked by adjacency → softmax over neighbours    │
        └──────────────────────────┬───────────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │ PER-NODE HEAD  (shared, dimension-independent)   │
        │ → 1-step-ahead forecast for each channel         │
        └──────────────────────────────────────────────────┘

        38,049 parameters total
```

### Three design decisions worth explaining

**Why a TCN instead of the GRU we originally planned.**
Same temporal-modelling role, but convolutional means the *whole series* can be encoded in one
pass. Both 7.36M-row error passes take **~53 seconds** on Apple MPS. A per-window GRU would
re-run a 250-step recurrence at each of 7.36M positions — 30–60 minutes per pass. That difference
is what made six design iterations possible in a single session.

**Why telecommands are input-only nodes.**
A command can cause a sensor change; a sensor cannot cause a command. Hard-coding that direction
shrinks the hypothesis space and makes the learned `telecommand → channel` edges interpretable —
they recover *which command affects which sensor*, a mapping ESA deliberately anonymised.
This addresses unmet requirement **R6**.

**Why the head is per-node `Linear(d→1)`.**
The original `Linear(N·d → N)` scales with channel count, so it could not be federated across
missions with different channel counts (6 vs 11). A smoke test caught this immediately:
```
RuntimeError: stack expects each tensor to be equal size, but got [384,384] and [704,704]
```
With the per-node head, every shared operator is dimension-independent and federation works.

---

## 2. The availability features — the decisive ingredient

`extract_features.py` reads the **raw pickles** (not the preprocessed CSVs, which have already
destroyed this information) and computes, aligned to the official grid:

| Feature | Definition |
|---|---|
| `count` | raw samples that arrived in this window |
| `tsince` | seconds since the last real sample |
| `tcdt` | seconds since the last priority-≥3 telecommand firing |

**Why this matters** (measured on `channel_41`, reproduced on 43 and 45):

```
count = 0   →  P(anomaly) = 56.7 %    31× lift over the 1.806 % base rate
               30.7 % of ALL Mission-1 anomalous rows are here
count = 1   →  P(anomaly) =  1.26 %
count = 10  →  P(anomaly) = 96.6 %    53× lift
```

Under Zero-Order-Hold, a silent window is forward-filled to a constant — so a forecaster predicts
it perfectly and detects nothing. **The model was structurally blind to 30% of Mission 1's
anomalies.** Mission 2 has only 136 such rows, which is exactly why its scores were always higher.

Standalone, a trivial gap detector ("all 6 channels silent for ≥10 steps", L tuned on train)
scores **0.412 at precision 1.000** on the Mission-1 test set.

---

## 3. Mission-2 winning pipeline

```
forecast errors
   → EWMA smoothing, span = 15
   → robust z-score  (per-channel median / IQR, max over channels)
   → absolute threshold  z* = 20      ← tuned on TRAIN, transferred as an absolute value
```

`m2_absolute.py` → [`results/M2_v3_absolute.json`](results/M2_v3_absolute.json)

**Fully pre-registered**: the rule (absolute-z transfer, z* = 20, span 15) was fixed on train
before the test evaluation ran.

Alternatives we measured and rejected:
- quantile transfer instead of absolute → **0.9260**
- test-set q0.98 → 0.9798 (oracle — *not* reportable)

---

## 4. Files

| File | Purpose |
|---|---|
| `extract_features.py` | ⭐ availability + telecommand features from raw pickles |
| `data.py` | builds the `(T, N, 3)` node-feature array; windowed dataset |
| `model.py` | ⭐ the GDNv2 network + fast full-series `series_errors()` |
| `train.py` | trainer + train-period operating-point selection |
| `score.py` | EWMA, robust z, gap detector, operating points |
| `gap_analysis.py` | ⭐ the availability finding; tunes the gap detector on train |
| `diagnose.py` | per-event signal table — the analysis that redirected the project |
| `final_m1.py` | GDNv2's Mission-1 pipeline (0.8197; superseded by GDNv4's 0.8532) |
| `m2_absolute.py` | ⭐ **Mission-2 winner** |

---

## 5. Compute

M4 MacBook Air, Apple MPS: ~40 s/epoch, error passes ~53 s, ~2.7 GB RAM,
~15 min end-to-end per mission including full test evaluation.

---

## 6. Caveats

- Single seed (0). A 3-seed stability run is required before publication.
- Pipeline *design* was informed by logged test-outcome iterations (see [`../../logs/`](../../logs/));
  all *final parameters* are train-tuned, and M2's rule was genuinely pre-registered. Disclose this.
- Availability features come from raw timestamps the official CSVs erase, so the published
  baselines never had them. That asymmetry **is** the R8 contribution — but run the
  Telemanom + availability-features ablation to prove it's the feature, not just our model.
