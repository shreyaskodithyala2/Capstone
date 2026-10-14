# Architecture — step by step, in execution order

Shapes are for **Mission 1 lightweight** (`n_ch=6`, `n_tc=11`, `N=17`, `T=7,364,161`)
unless stated. Full-set numbers in brackets: `[M1_full: n_ch=76, N=87]`.

---

## Stage 0 — the grid (ESA's preprocessing, not ours)

Raw telemetry is irregularly sampled. ESA resamples onto a fixed grid
(**30 s** for M1, **18 s** for M2) by **Zero-Order-Hold**: within each bucket,
last-sample-wins; if no sample arrives, the previous value is carried forward.

Output: `84_months.{train,test}.csv`, 175 columns for M1
(`timestamp | channel_1..76 | telecommands | is_anomaly_*`).

**This step destroys information** — see Stage 1.

---

## Stage 1 — availability features  (`src/extract_features.py`)

ZOH makes "a fresh sample arrived" and "nothing arrived, value carried forward"
indistinguishable in the CSV. We recover it from the **raw pickles**.

For each channel, with `raw_ns` = raw sample times and `g_ns` = grid times:

```python
idx      = np.searchsorted(raw_ns, g_ns, side="right")
prev_idx = np.searchsorted(raw_ns, g_ns - grid_step_ns, side="right")
count    = idx - prev_idx                       # samples in this bucket   -> uint16
tsince   = (g_ns - raw_ns[idx-1]) / 1e9         # seconds since last sample -> float32
```

For each priority-≥3 telecommand: `tcdt` = seconds since its last firing (cap 1e7).

Written to `{mk}_{split}_feats.npz`:
`count (T, n_ch)`, `tsince (T, n_ch)`, `tcdt (T, n_tc)`.

**All three are label-free** — computable onboard in real time. This is the R8
contribution.

---

## Stage 2 — node feature tensor  (`src/dataset.py::build_arrays`)

```python
vals = _vals(mk, split)                                  # (T, n_ch) float32
mu   = vals.mean(0)
sd   = np.maximum(vals.std(0), m["sd_floor"]) + 1e-8     # per channel
vals = (vals - mu) / sd

X = np.zeros((T, n_ch + n_tc, 3), np.float32)            # (T, N, 3)
X[:, :n_ch, 0] = vals                                    # normalised value
X[:, :n_ch, 1] = (count == 0)                            # gap flag
X[:, :n_ch, 2] = np.log10(1.0 + tsince) / 7.0            # log staleness, ~[0,1]
X[:, n_ch:, 0] = np.exp(-tcdt / 1800.0)                  # TC recency, 30-min decay
```

The graph has **N = n_ch + n_tc** nodes:
- **channel nodes** carry 3 features
- **telecommand nodes** carry recency in slot 0, zeros elsewhere — they are
  **input-only exogenous** nodes

`sd_floor` is `0.0` for the lightweight configs (bit-identical to published runs)
and `1e-3` for `*_full`, because M1 has 4 channels with sd 1e-7–1e-6 whose own-sd
normalisation would amplify a 2.7e-4 blip to ±2700 and dominate the MSE loss.

`is_anomaly` is **never** an input. Train-split labels are supervision for the
detector thresholds only.

---

## Stage 3 — the encoder: causal dilated TCN  (`src/model.py::TCN`)

```python
dil   = [1, 2, 4, 8, 16, 32]           # 6 layers
convs = [Conv1d(3 -> 32, k=5, dilation=1),
         Conv1d(32 -> 32, k=5, dilation=2), ... dilation=32]
proj  = Conv1d(32 -> 64, k=1)          # ONE 1x1 conv at the end

for conv, p in zip(convs, pads):
    h = F.relu(conv(F.pad(h, (p, 0)))) + (h if shapes match else 0)
return proj(h)
```

Three things to be precise about:

1. **Causality** comes from asymmetric padding: `F.pad(h, (p, 0))` pads only the
   left by `p = (k-1) * dilation`. Position `t` never sees `t+1`.
2. **The residual is an identity skip**, added *after* ReLU. Layer 0 is skipped
   (3 channels in, 32 out — shapes differ), layers 1–5 get it.
3. **There is no feed-forward block between layers.** `conv → ReLU → add`, six
   times. `proj` is a position-wise linear map applied once at the end.

**Receptive field:** `RF = 1 + (k-1) * sum(dil) = 1 + 4*63 = 253`

| layer | dilation | adds | cumulative RF |
|---|---|---|---|
| 0 | 1 | 4 | 5 |
| 1 | 2 | 8 | 13 |
| 2 | 4 | 16 | 29 |
| 3 | 8 | 32 | 61 |
| 4 | 16 | 64 | 125 |
| 5 | 32 | 128 | **253** |

253 steps = **2.11 h** on M1's 30 s grid (1.27 h on M2's 18 s). Telemanom uses
250 — deliberately matched.

**The encoder is node-agnostic.** `encode()` does
`x.reshape(B*N, T, F).transpose(1, 2)`, so all `B*N` node-series pass through the
*same* weights. That is why it can be federated across missions with different
channel counts.

---

## Stage 4 — the graph  (`src/model.py::GDNForecaster`)

### 4a. Structure learning — `get_adj()`

```python
v   = self.embedding.weight                      # (N, 64), learned
vc  = F.normalize(v[:n_ch], dim=1)
sim = vc @ vc.t() - eye * 9e15                   # cosine, self excluded
A[:n_ch, :n_ch].scatter_(1, sim.topk(k, 1).indices, 1.0)   # k = min(n_ch-1, 10)
A[:n_ch, n_ch:] = 1.0                            # every channel sees every TC
A += eye
```

Each channel attends to its **top-10 most similar channels** (by embedding cosine)
plus all telecommand nodes plus itself. The `tc -> ch` block stays zero: a sensor
can never influence a command.

This mask is **static** — it depends only on `embedding.weight`, so at eval it is
cached.

### 4b. Attention — `predict_positions(h_pos)`, `h_pos: (P, N, 64)`

```python
g     = torch.cat([v, h_pos], dim=2)                    # (P, N, 128)
e     = leaky(a1(g) + a2(g).transpose(1, 2))            # (P, N, N)
e     = e.masked_fill(A == 0, -inf)
alpha = softmax(e, dim=2)
z     = relu(alpha @ h_pos)                             # (P, N, 64)
gz    = torch.cat([v * z, z], dim=2)                    # (P, N, 128)
out   = head(gz).squeeze(-1)[:, :n_ch]                  # (P, n_ch)
```

The **decomposition** matters for memory: writing
`e_ij = LeakyReLU(a1·g_i + a2·g_j)` as the sum of two `(P, N, 1)` vectors
broadcast against each other avoids ever materialising `(P, N, N, 128)`.
Even so `e` is `(P, N, N)` — 19 MB at N=17, **188 MB at N=87** — which is why
`series_errors` auto-sizes its chunk by an element budget.

Note `g` contains `h_pos`. **The attention weights therefore already vary with
time.** Only the top-k *mask* is static. A per-window dynamic mask is unclaimed.

`head = Linear(128 -> 64) -> ReLU -> Linear(64 -> 1)`, applied per node, so it is
independent of channel count — also required for federation.

### Parameter budget

| | M1 lite | M2 lite | M1 full |
|---|---|---|---|
| `embedding` (**local**) | 1,088 | 960 | 5,568 |
| TCN + a1/a2 + head (**shared**) | 36,961 | 36,961 | 36,961 |
| total | 38,049 | 37,921 | 42,529 |

The shared block is **identically 36,961** in every configuration. That is the
federation story in one number.

---

## Stage 5 — training  (`src/train.py::train_model`)

```python
tr_starts = np.arange(0, vsplit - seg - 1)       # vsplit = 85% of T, chronological
x, y = sample_batch(X, tr_starts, seg=512, ...)  # x: (24, N, 512, 3)
loss = MSELoss(model.forward_segments(x), y)     # y: (24, 512-253, n_ch)
```

`forward_segments` encodes the whole 512-step segment, keeps positions
`RF-1 .. L-2`, and predicts the value at `t+1` for each. One segment yields
`512 - 253 = 259` supervised positions, so a batch of 24 gives
`P = 6,216` prediction points.

Adam, lr 1e-3, 300 batches/epoch, ≤30 epochs, early stop patience 4 on a
**chronologically held-out** last 15 %. Horizon is **1 step** (Telemanom uses 10 —
an open gap).

---

## Stage 6 — full-series error pass  (`src/model.py::series_errors`)

```python
for a in range(RF, T, enc_chunk):
    seg   = X[a-RF : b]                       # (1, N, T', 3)
    h     = model.encode(seg)[0]              # (N, T', 64)
    hp    = h[:, RF-1 : RF-1+(b-a)].permute(1, 0, 2)
    preds = predict_positions(hp)             # chunked over P
    errs[a:b] = |preds - X[a:b, :n_ch, 0]|
```

Output `errs: (T, n_ch) float32`, zeros in the first 253 rows.

Because the encoder is convolutional, **every position's encoding is computed
once** — a full 7.36 M-row pass takes ~53 s on MPS, ~118 s at 76 channels on a
T4. A per-window GRU would re-run a 250-step recurrence at each of 7.36 M
positions: 30–60 min. This is the single reason iteration was feasible.

---

## Stage 7 — error conditioning  (`src/run_full_channels.py::phase`)

```python
gap_any = (cnt == 0).all(1)                   # ALL channels silent
ec      = zero_gap_margin(err, gap_any, 30)   # zero errors in gaps + 30 steps
es5     = ewma(ec, 5)                         # branch A input
Zl      = multi_span_z(ec, [1,3,15,60,240], block=2880)   # block-local z
Zg      = multi_span_z(ec, [1,3,15,60,240], block=len(ec))# global z
tcw     = (tcdt.min(1) <= 7200.0)             # within 2 h of any priority-3 TC
```

**Why zero the gaps:** on re-acquisition after an outage the forecast is wildly
wrong and `z` reaches ~1149. Those spikes poison Telemanom's sorted-peak logic.
M1 test has 16,676 gap runs vs 3,196 in train — justified on **input statistics
alone**, no labels.

`block_z` normalises by a *local* median/IQR over 2880-step blocks (24 h on M1),
linearly interpolated, with the IQR floored at that channel's own 25th percentile
so a quiet block cannot manufacture huge z from numerical noise.
`multi_span_z` takes the max over 5 EWMA spans — a single span (5) once
caught 0 of 12 events of length 6–60.

---

## Stage 8 — four detectors, OR-ed per channel

| | branch | mechanism |
|---|---|---|
| **A** | Telemanom prune | dynamic threshold `find_epsilon` + `prune_anoms` on `es5` |
| **B** | multi-span z | `Z > thr`, `thr` = train quantile `q`, global or **per-channel** |
| **C** | gap detector | per-channel `count==0` runs ≥ `L_c` (M1 only) |
| **D** | TC-gated | `Z > thr*boost` **and** `tcw` — a lower bar near a command |

**A, in detail.** `find_epsilon` sweeps `z ∈ [2.5, 12)` and picks the threshold
maximising `(mean%drop + sd%drop) / (len(E_seq)^2 + len(i))` subject to ≤5
sequences and <50 % of points flagged. Then `prune_anoms` sorts candidate
sequences by peak error descending, appends the non-anomalous background max, and
removes the **trailing run of sequences whose relative drop to the next is < p**
— i.e. it discards the *marginal* detections that are indistinguishable from
background, keeping only the ones that stand out. Removing this drops EW
precision **0.999 → 0.444**.

**C, in detail.** A flat `L=10` fails at full scale: `channel_61` is sampled every
~15 min (median `count==0` run of 29 steps on a 30 s grid), so a 10-step silence
is *normal* for it and the detector flags it 96.67 % of the time. Fixed with

```python
L_c = max(L, ceil(mult * median_zero_run_c))     # cadence fitted on TRAIN
```

Flagged rows 96.94 % → 7.91 %; channels flagging >50 %: 3 → 0.
**Confirmed in the full run:** `gapmult=None` scores **0.0000**, `gapmult=6.0`
scores 0.8814 on train.

---

## Stage 9 — morphology

```python
P = open_runs2d(close_runs2d(P, close), minlen)
```

- **closing** bridges 0-gaps ≤ `close`, merging fragments of one event — this is
  what alarming precision (aspect 3) rewards
- **opening** drops runs shorter than `minlen`, killing isolated blips

Implemented on a run table, **not** `scipy.ndimage`: flat-structuring-element
morphology is O(T·k) in the element width, the run form is O(#runs) and 3× faster
at these sizes.

---

## Stage 10 — tuning and scoring

Grid axes: `p`, `zmode`, `tmode`, `q`, `close`, `minlen`, `gapL`, `gapmult`,
`boost`. **Everything is tuned on TRAIN labels only**, the winner is written to
`*_preregistered.json`, train arrays are freed, and the test period is then
evaluated **exactly once** with that frozen config.

```
ESAScores(betas=0.5, select_labels={"Category": ["Rare Event", "Anomaly"]})
Pre = TPe/(TPe+FPe) * (1 - FPt/Nt)
```

`(1 - FPt/Nt)` penalises false-positive **duration**; β=0.5 weights precision 4×.
Only the 58 `Target=YES` channels are scored — the official
`infer_anomaly_types.py` never labels the rest.

---

## Known weak points

| aspect | ours | published best |
|---|---|---|
| 1a event-wise F0.5 | **0.8532** | 0.786 |
| 2b channel-aware | 0.4740 | 0.837 |
| 3 alarming precision | 0.7955 | 0.875 |
| 4 ADTQC | **0.1864** | 0.901 |
| 5 affiliation | 0.6812 | 0.734 |

ADTQC is diagnosed: `Nb_Before=23` vs `Nb_After=12` — closing and the TC boost
make alarms fire *before* the labelled onset. Fix: emit the first index of each
merged run.
