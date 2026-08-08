# 3 · The GDN model and the federated design

---

## Part A — Why GDN

The core problem, shown on a real row from our data:

```
Normal:   ch41 0.803  ch42 0.793  ch43 0.774  ch44 0.786  ch45 0.814  ch46 0.772
ANOMALY:  ch41 0.970  ch42 0.933  ch43 0.000  ch44 0.958  ch45 0.975  ch46 0.928
```

Channel 43 reads 0.000. Is that outside its normal range? **No** — the channel is scaled 0–1 and
legitimately visits zero. A threshold alarm stays silent.

But five sensors bolted to the same hardware jumped *up* while the sixth went to *zero*. The
anomaly lives in the **relationship**, not in any single number.

> **GDN's question:** not *"is this value strange?"* but *"does this value still fit what the
> other sensors are doing?"*

This is the right family for ESA-ADB because **83 of Mission 1's 200 events are `Locality =
Local`** — normal-looking values that are wrong *in context*. And we proved point detectors fail
here: official iForest scores **0.00007**, HBOS **0.00014**, PCC **0.00013**.

---

## Part B — GDN's four stages

Reference: Deng & Hooi, AAAI 2021 (`papers/models/GDN_Deng-Hooi_AAAI2021.pdf`).
Our implementation: [`src/gdn.py`](../src/gdn.py).

### 1. Sensor embedding

Each of the `N` sensors gets a learnable **embedding vector** of dimension `d` — a table of shape
`N × d`, randomly initialised, updated by backpropagation.

These are **static node features**: they describe the sensor's *character*, not its reading.
Standard GNNs share identical parameters across nodes, which is wrong when a temperature sensor
and a voltage sensor sit in the same graph.

> **Common misconception:** these embeddings are **never updated by message passing**. They are
> parameters, changed only by the optimiser. What gets message-passed is the *telemetry window*.

### 2. Graph structure learning

The topology is **not supplied** — ESA anonymised everything, so there is no wiring diagram.
GDN infers it: L2-normalise the embeddings → pairwise **cosine similarity** → **top-k** per node →
binary **adjacency matrix**.

⚠️ The similarity matrix is **symmetric by construction**. GDN cannot express "A drives B but not
the reverse."

### 3. Graph-attention forecasting

- Window → **shared linear transform** → node features
- **Concatenate each node's embedding with its transformed window** (GDN's departure from GAT)
- **Attention coefficients** over neighbours, softmax restricted to the adjacency
- **Message passing**: aggregate neighbour features by those weights
- Gate by the node's own embedding → MLP head → **predicted next value**

Trained with MSE on nominal data — **semi-supervised**.

### 4. Graph deviation scoring

- Per-sensor **forecast residuals**
- **Robust normalisation** using median and IQR — deliberately not mean/σ, because those are
  distorted by the very anomalies being detected
- Score = **max across sensors** (not mean), so a single-channel fault isn't diluted
- Per-sensor scoring **localises which channel broke**

**On our real anomaly:** seeing channels 41/42/44/45/46 at 0.93–0.97, the model confidently
predicts channel 43 ≈ 0.95. Reality: 0.000. Massive residual → detected. The threshold detector
missed it entirely.

### Order of operations (one forward pass)

```
A  read embedding table                    (current parameter values)
B  BUILD GRAPH: normalise → cosine → top-k → adjacency A
C  transform input window → node features
D  attention scores over pairs: concat(embedding, features)
E  MASK by A, then softmax
F  MESSAGE PASSING: aggregate neighbour features
G  gate by embedding → MLP → prediction
H  loss → backprop → optimiser updates the embeddings   ← the ONLY thing that changes them
```

Graph construction (B) must precede message passing (F) — you can't aggregate over neighbours
until you know who they are. `get_adj()` is called *inside* `forward()`, so the graph is rebuilt
every batch from the current embeddings; it stabilises as training converges and is fixed
afterwards.

---

## Part C — Our implementation and its hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `N` (nodes) | **6** (M1), **11** (M2) | channels 41–46 / 18–28 |
| `d` (embedding dim) | **64** | *Untuned default.* 6 points in 64-D is arguably oversized — worth ablating at 8/16/32 |
| `window` | **100** steps (50 min at 30 s) | Telemanom uses **250** — likely a real gap |
| prediction horizon | **1** step | Telemanom predicts **10** — likely a real gap |
| `topk` | **N−1** | ⚠️ **see below** |
| epochs | 16 | Telemanom trains up to 1000 with early stopping |
| optimiser | Adam, lr 1e-3, batch 256 | |
| parameters | ~11.3 K (M1), ~11.6 K (M2) | after the federated redesign |

### ⚠️ Known issue: graph learning is effectively disabled

```python
self.topk = topk if topk is not None else (n_nodes - 1)
```

We never passed `topk`, so **k = N−1 = 5** for Mission 1 — meaning **every node connects to every
other node**. We verified the adjacency matrix comes out completely filled.

**Consequence:** top-k selection is vacuous, so the embeddings have **zero influence on
topology**. GDN's headline contribution — the learned graph — currently contributes nothing
structural. (Embeddings still affect *attention weights*, so they aren't useless.)

**Cheap fix worth running:** sweep k = 2, 3, 4, 5 (M1) and 3, 5, 8, 10 (M2). STGLR reports this
ablation matters. Either outcome is publishable — if a sparser graph doesn't help, that
*supports* the finding that channels 41–46 are too correlated for topology to carry information.

---

## Part D — The cross-mission federated design

**The mechanical problem.** Mission 1 has 6 channels, Mission 2 has 11. You cannot average their
weights — the embedding tables are different shapes (`6×64` vs `11×64`) and the sensors mean
different physical things. Standard FedAvg/FedProx **requires identical parameter shapes**.

**The fix — partial parameter sharing (FedPer-style):**

| Parameter group | Shape depends on N? | Federated? |
|---|---|---|
| **Sensor embeddings** (`6×64` vs `11×64`) | ✅ yes | ❌ **stays local** (personalised) |
| **Shared operators** — window transform, attention vector, output head | ❌ no | ✅ **federated** |

Intuition: embeddings encode *"who these specific sensors are"* (mission-specific, private).
Operators encode *"how to do graph-attention forecasting in general"* (transferable). Share the
skill, keep the identities local.

### A redesign this forced

The original output layer was `Linear(N·d → N)` — its shape scales with the channel count, so it
**could not be federated**. Our smoke test caught this immediately:

```
RuntimeError: stack expects each tensor to be equal size,
but got [384, 384] at entry 0 and [704, 704] at entry 1
```

Fixed by making the head **per-node**: `Linear(d → 1)` applied to each sensor independently. Now
every shared operator is dimension-independent:

```
shared keys: W.weight, W.bias, att.weight, out.0.weight, out.0.bias, out.2.weight, out.2.bias
all shapes match across N: True
```

### The training loop

```
Client 1 = Mission 1 GDN (6 embeddings + shared operators)
Client 2 = Mission 2 GDN (11 embeddings + shared operators)

for round r = 1..R:
    each client trains its FULL GDN locally for a few epochs
        FedProx: add μ‖w − w_global‖² so clients don't drift apart
    each client uploads ONLY its shared-operator weights (never embeddings)
    server aggregates with TRIMMED MEAN (discards outlier/faulty updates)
    server broadcasts the new shared weights back

after the rounds:
    each client fine-tunes locally for a few epochs   ← PERSONALIZATION, critical
```

`--mu 0.01` (FedProx), `--trim 0` (with only 2 clients, trimming would discard everything;
enable when scaling to more clients).

### Personalization was decisive

Without the local fine-tuning phase, federation **hurt** Mission 1:

| | Solo | Federated (no fine-tune) | Federated (+ fine-tune) |
|---|---|---|---|
| **M1** | 0.1765 | 0.1370 ❌ **−0.040** | **0.1892** ✅ **+0.013** |
| **M2** | 0.7739 | 0.7820 (+0.008) | **0.8050** ✅ **+0.031** |

*(oracle-threshold scoring, used for internal comparison only)*

This is the classic **non-IID** failure: the shared operators get dragged toward Mission 2 (which
has far more anomaly events driving gradients), degrading Mission 1. A short local re-specialisation
phase fixes it.

**Independent corroboration:** Atefrad & Karami (`papers/federated/`) report the same pattern —
their fine-tuning moved F0.5 from 0.74 → 0.86. Two teams converging on
personalization-after-aggregation is a solid result to report.

---

## Part E — The code

| File | Purpose |
|---|---|
| [`src/gdn.py`](../src/gdn.py) | The GDN model. `shared_state()` / `load_shared_state()` implement the federated split |
| [`src/common.py`](../src/common.py) | Data loading, windowing, deviation scoring, the two scorers |
| [`src/federated.py`](../src/federated.py) | Solo vs federated experiment: FedProx, trimmed mean, personalization |
| [`src/train_m1.py`](../src/train_m1.py) | Single-mission training (Stage A) |
| [`src/official_score.py`](../src/official_score.py) | ⭐ **Byte-faithful replica of the official TimeEval scoring** — validated to reproduce 0.9487 |
| [`src/telemanom_prune.py`](../src/telemanom_prune.py) | Faithful port of Telemanom's dynamic thresholding + pruning |
| [`src/tune_best.py`](../src/tune_best.py) | Threshold / pruning-parameter sweep on validation |
| [`src/official_repro.py`](../src/official_repro.py) | Runs official baselines through Docker + TimeEval |
| [`src/make_light_csvs.py`](../src/make_light_csvs.py) | Channel-trimmed CSVs so Docker containers fit in RAM |

### `official_score.py` — why it matters

Rather than Dockerising every model variant, we replicated TimeEval's scoring exactly: same
`y_true` construction (events fully inside the test range, merged with `anomaly_types`, **not**
deduplicated), same `ESAScores` call.

**Validated:** re-scoring the official `subsequence_if` container output through our replica gives
`EW_F_0.50 = 0.9487`, `precision = 0.9509`, `recall = 0.94` — **identical to the harness.** So any
model we write can be scored comparably against published baselines without a container.

### Running it

```bash
conda activate timeeval

# single mission
python src/train_m1.py --w 100 --embed 64 --epochs 30

# solo vs federated
python src/federated.py --rounds 6 --local_epochs 2 --ft_epochs 4 --mu 0.01

# best comparable numbers (validation-tuned, no test peeking)
python src/tune_best.py
```

Runs comfortably on an M4 MacBook Air (Apple **MPS**): ~600 k windows, ~23 s/epoch, ~1 GB RAM,
~10–20 min including full test-set evaluation.

> Paths are currently absolute (`/Users/Shreyas/...`) in `common.py` and `official_score.py`.
> **Edit `PREP` and the mission `raw` paths** before running on another machine.
