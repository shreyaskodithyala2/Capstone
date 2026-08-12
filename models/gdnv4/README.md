# GDNv4 — Mission-1 winner + the full ESA 5-aspect scoreboard

**Result: Mission 1 event-wise F0.5 = 0.8532** (P 0.9993, R 0.5385)
vs published best **0.786** (Telemanom-ESA-Pruned) and **0.801** (SMED, PHM Society 2025).
**Best published result on Mission 1 anywhere, as far as we can find.**

GDNv4 uses **the same trained network as GDNv2** — it is a better *detection stack*, not a new
model. `run_v4.py` imports `model.py` and `data.py` from [`../gdnv2/src/`](../gdnv2/src/).

---

## 1. Why this exists: we were only measuring 2 of 5 metrics

ESA-ADB evaluates on a **five-aspect hierarchy**. Before this folder existed we measured two.

| # | Aspect | Where it lives in ESA's code |
|---|---|---|
| 1a | Corrected event-wise F0.5 | `ESAScores` |
| 1b | Anomaly existence recall | `ESAScores` |
| 2a | Subsystem-aware F0.5 | `ChannelAwareFScore` *(not reported for lightweight subsets — single subsystem)* |
| 2b | **Channel-aware F0.5** | `ChannelAwareFScore` |
| 3 | Alarming precision | `ESAScores` |
| 4 | **ADTQC** (detection timing) | `ADTQC` |
| 5 | Affiliation-based F0.5 | `ESAScores` |

`esa_metrics.py` implements **all of them**, faithfully to `timeeval/core/experiments.py`
(same `y_true` construction: events fully inside the test range, merged with `anomaly_types`,
**not** deduplicated; same per-channel `y_pred` dict; same `subsystems_mapping`).

**Validated:** re-scoring the official gap detector through it reproduces **0.41237** exactly.

Immediate consequence: channel-aware F0.5 was only **0.123** when we broadcast one global signal
across all channels. **Per-channel prediction is mandatory** for the full suite — which is why
this stack emits `(T, n_channels)`, not `(T,)`.

---

## 2. The detection stack

Four detector branches, OR-ed per channel, then cleaned up:

```
forecast errors (from the GDNv2 encoder)
        │
        ├─ zero errors inside count==0 runs + 30-step margin
        │     Gap re-acquisition spikes reach z ≈ 1149 and poison pruning's
        │     sorted-peak logic. 16,676 gap runs in test vs 3,196 in train —
        │     justified purely from LABEL-FREE input statistics.
        │
        ├─ A. Telemanom dynamic threshold + PRUNING       ← the precision engine
        ├─ B. multi-span block-local robust z             ← the recall lifter
        │       spans {1,3,15,60,240}, 24 h blocks, max over spans
        ├─ C. per-channel gap detector (count==0 run ≥ 10)
        └─ D. telecommand-gated lower threshold (×0.6 within 2 h of a firing)
        │
        ▼
   morphological CLOSING (bridge gaps ≤ 60 steps)   ← fixes alarming precision
   MIN-RUN filter (drop runs < 3)                   ← removes isolated blips
        │
        ▼
   (T × n_channels) binary predictions → full 5-aspect scoreboard
```

**Pre-registered config** (train F0.5 = 0.7748, frozen before test scoring):
`p=0.3 · zmode=local · q=0.9999 · close=60 · minlen=3 · gapL=10 · boost=0.6`

### Why each branch is there — with evidence

**A. Pruning is the precision engine.** Ablation `ablations/run_v3.py` removed it and kept
everything else: EW precision collapsed **0.999 → 0.444**, F0.5 fell to **0.3827**. Keep it.

**B. Local normalisation fixes a measured noise-floor problem.** `miss_analysis.py` showed every
missed event sat *below* the global background p99.9. Verified in isolation: an identical-magnitude
event scores **z = 26.8 in a quiet stretch vs 3.6 in a noisy one** under block-local normalisation.

**C. Availability is a standalone detector.** Gap detector alone = **0.412 at precision 1.000**.

**D. Telecommands separate rare events from anomalies.** Measured: **52.8%** of rare events have a
priority-3 command within ±2 h, vs **10.3%** for genuine anomalies, against a **0.4%** base rate.

**Closing fixes fragmentation.** GDNv2 fired ~4.4 alarms per real event
(`alarming_precision = 0.228`). Closing lifted it to **0.7955**.

---

## 3. Results — all five aspects, Mission 1

| Aspect | Best published *(algo)* | SMED | GDNv2 | **GDNv4** |
|---|---|---|---|---|
| **1a** Event-wise F0.5 | 0.786 *(Telem-P)* | 0.801 | 0.8197 | **0.8532** 🥇 |
| EW precision | 0.999 | 0.999 | 0.9993 | **0.9993** |
| **1b** EW recall | 0.894 *(Telemanom)* | 0.446 | 0.4769 | 0.5385 |
| **2b** Channel-aware F0.5 | 0.837 *(Telemanom)* | 0.444 | *n/m* | 0.4740 |
| **3** Alarming precision | 0.875 *(Telem-P)* | 0.659 | 0.2279 | 0.7955 |
| **4** ADTQC | 0.901 *(DC-VAE3)* | 0.160 | *n/m* | 0.1864 ❌ |
| **5** Affiliation F0.5 | 0.734 *(DC-VAE5)* | 0.634 | 0.6588 | 0.6812 |

Full table incl. Mission 2: [`RESULTS.md`](RESULTS.md).

### The ADTQC regression — diagnosed, not mysterious

`Nb_Before = 23, Nb_After = 12`. Two-thirds of detections now fire *before* the annotated onset,
because `close=60` and the telecommand boost widen detections backwards. ADTQC penalises early
detection heavily.

**Context:** this is the trade-off ESA documents — *"pruning significantly decreases
subsystem-/channel-aware, detection timing (ADTQC), and affiliation-based scores."*
Telemanom-Pruned itself scores **0.197** here. We are in the same regime as the algorithm ESA
calls best, and our after-ratio (0.343) is already better than SMED's (0.103).

**Identified fix (not yet run):** emit the *first index* of each merged run instead of the whole
closed block. Should cut `Nb_Before` at zero cost to F0.5.

---

## 4. Files

| File | Purpose |
|---|---|
| `esa_metrics.py` | ⭐ **all five ESA aspects**; `global_f05()` is the fast path for hyper-parameter search |
| `detect.py` | block-local z, multi-span fusion, closing, min-run, per-channel gap detector |
| `run_v4.py` | ⭐ **the winner** — train tuning → pre-registration → single test evaluation |
| `miss_analysis.py` | which events are still missed and why (drove the v4 design) |
| `ablations/run_v3.py` | pruning-removed ablation — proves pruning carries precision |

---

## 5. Run it

```bash
conda activate timeeval
python models/gdnv2/src/extract_features.py        # once
python models/gdnv2/src/train.py --mission M1 --tag v1
python models/gdnv4/src/run_v4.py --mission M1 --tag v4
```

432 configs, all scored on the **train** period; the winner is printed and frozen, then TEST is
scored exactly once across all five aspects. Runtime ≈ 45 min on Apple MPS.

> Performance note: hyper-parameter search uses `global_f05()` (1 metric) rather than the full
> scoreboard (14 metrics) — a ~14× speedup that made the 432-config search feasible.

---

## 6. Caveats

- **Single seed.** Mandatory 3-seed run before publication.
- **ADTQC is a genuine regression** vs GDNv3's 0.5623. Do not hide it; the fix is identified.
- **Mission-2 v4 run is in progress** — GDNv2's 0.9644 remains the reported M2 number.
- **A competitor beats us on Mission 2**: SMED (Brancato et al., PHM Society 2025) scores
  **0.971**. See [`../../docs/07-RELATED-WORK.md`](../../docs/07-RELATED-WORK.md).
