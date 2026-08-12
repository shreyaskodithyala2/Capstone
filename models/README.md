# Models — GDNv2 and GDNv4

Two folders, **one neural network**. GDNv4 is a better *detection stack* running on the encoder
trained by GDNv2. Understanding that split is the key to reading this code.

```
                        ┌───────────────────────────────┐
                        │  SHARED NEURAL NETWORK        │
                        │  gdnv2/src/model.py           │
                        │  dilated TCN + graph attention│
                        │  38k params · RF = 253 steps  │
                        └───────────────┬───────────────┘
                                        │ forecast errors (T × N)
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
        ┌───────────────────────────┐   ┌──────────────────────────────┐
        │ GDNv2 detection stack     │   │ GDNv4 detection stack        │
        │ EWMA(15) → robust z       │   │ Telemanom prune              │
        │        → absolute z* = 20 │   │ + multi-span block-local z   │
        │                           │   │ + gap detector               │
        │                           │   │ + closing / min-run          │
        │                           │   │ + telecommand-gated threshold│
        └───────────┬───────────────┘   └───────────────┬──────────────┘
                    ▼                                   ▼
          MISSION 2 WINNER                     MISSION 1 WINNER
          EW F0.5 = 0.9644                     EW F0.5 = 0.8532
          (published best 0.949)               (published best 0.786)
```

---

## Which folder does what

| Folder | Contains | Winner for |
|---|---|---|
| [`gdnv2/`](gdnv2/) | Feature extraction, data pipeline, **the model**, training loop, the gap analysis, the per-event diagnosis, and the M2 detection pipeline | **Mission 2** |
| [`gdnv4/`](gdnv4/) | The **full ESA 5-aspect scoreboard**, the advanced detection stack, and the M1 tuning/eval runner | **Mission 1** |

`gdnv4/src/run_v4.py` does `sys.path.insert(...)` on `gdnv2/src` and imports `model.py`, `data.py`.
**You must keep both folders.**

---

## Why two different detection stacks?

This is a genuine finding, not an accident. We measured the same pruning step on both missions:

| Detection stack | Mission 1 | Mission 2 |
|---|---|---|
| Telemanom dynamic threshold + pruning | **0.8532** ✅ | **0.014** ❌ |
| Absolute robust-z threshold | 0.137 ❌ | **0.9644** ✅ |

**The missions need opposite strategies.**

- **Mission 1** anomalies are sharp, rare, and buried in a noisy background full of gap-recovery
  spikes (background z reaches 1149). Pruning's "keep only detections that stand decisively above
  the crowd" logic is exactly right; a plain threshold drowns in false positives.
- **Mission 2** rare events are subtle and numerous (613 of 644 events). Pruning discards almost
  all of them — recall collapses to 0.003.

This is why the federated design personalises the **detection policy**, not just model weights.
See [`../docs/06-ROADMAP.md`](../docs/06-ROADMAP.md).

---

## Reproduction order

```bash
conda activate timeeval

# 1. availability features from the RAW pickles (~3 min)
#    the preprocessed CSVs have already destroyed this information
python models/gdnv2/src/extract_features.py

# 2. train the shared encoder (~15 min per mission on Apple MPS)
python models/gdnv2/src/train.py --mission M1 --tag v1
python models/gdnv2/src/train.py --mission M2 --tag v1

# 3. Mission-1 winner  → 0.8532
python models/gdnv4/src/run_v4.py --mission M1 --tag v4

# 4. Mission-2 winner  → 0.9644
python models/gdnv2/src/m2_absolute.py
```

Optional analyses that produced the key findings:

```bash
python models/gdnv2/src/gap_analysis.py    # the availability finding; gap detector alone = 0.412 @ P=1.0
python models/gdnv2/src/diagnose.py        # per-event signal table (the turning point)
python models/gdnv4/src/miss_analysis.py   # which events the final pipeline still misses, and why
```

---

## Ablations (kept deliberately)

`ablations/` folders hold intermediate runs. They are **not dead code** — each one proves
something the final design depends on.

| Ablation | Result | What it proves |
|---|---|---|
| `gdnv4/ablations/run_v3.py` | M1 EW F0.5 **0.3827** | Removing Telemanom pruning collapses precision 0.999 → 0.444. **Pruning is the precision engine.** It also lifted alarming precision 0.228 → 0.640, which is why v4 keeps both. |
| `gdnv2/ablations/iterate2.py` | 0.297 | Gap-cleaned thresholding alone is insufficient — background spikes persist beyond the gap margin |
| `gdnv2/ablations/iterate3.py` | 0.8371 (grid) | Cleaning + pruning together separate spike-from-spike; led directly to the final design |
| `gdnv2/ablations/iterate_m2.py` | 0.9260 | Quantile threshold transfer is too conservative on M2; motivated the absolute-z rule |

---

## Experimental protocol (identical for both models)

1. Model trained on the **train split only** (pre-2007 for M1, pre-Oct-2001 for M2).
2. **Every** detection hyper-parameter selected by official F0.5 on the **train period**.
   Legitimate: ESA-ADB ships label columns in the train CSVs and lists "learn from known
   anomalies" (R4) as a desired capability.
3. Winning config **printed and frozen** before test scoring.
4. **One** test evaluation, reporting all five ESA aspects.

No test labels were used to select any reported parameter. Full audit trail in [`../logs/`](../logs/).
