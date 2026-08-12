# 5 · How ESA-ADB is evaluated — and how we replicate it

Getting this right is the single most important thing in the project. A wrong metric makes every
other number meaningless.

---

## ESA's five-aspect hierarchy

ESA-ADB does **not** score with one number. It defines five aspects, checked in priority order:

| # | Aspect | Question it answers | Class in ESA's code |
|---|---|---|---|
| **1a** | Corrected event-wise F0.5 | Did you find the events without crying wolf? | `ESAScores` |
| **1b** | Anomaly existence recall | What fraction of events did you find at all? | `ESAScores` |
| **2a** | Subsystem-aware F0.5 | Did you blame the right subsystem? | `ChannelAwareFScore` |
| **2b** | Channel-aware F0.5 | Did you blame the right *channel*? | `ChannelAwareFScore` |
| **3** | Alarming precision | One alarm per event, or forty? | `ESAScores` |
| **4** | ADTQC | Did you detect at the right *time*? | `ADTQC` |
| **5** | Affiliation-based F0.5 | How close were you, in time? | `ESAScores` |

> ⚠️ **Subsystem-aware is not reported for lightweight subsets.** ESA states it explicitly: those
> channels all come from a single subsystem, so the metric is degenerate. Don't compare it.

Aspect 1a is the headline number in the paper's Table 4. Aspects 2–5 live in Appendix D
(Tables 16 and 17) and ESA calls them "lower priority metrics" — but they are still part of the
benchmark, and **no published algorithm wins all of them.**

---

## Aspect 1a in detail — the corrected event-wise F0.5

```
Pre  = TPe / (TPe + FPe)  ×  (1 − FPt / Nt)
Fβe  = (1 + β²) · Pre · Rece / (β² · Pre + Rece)          β = 0.5
```

Two things make this brutal, and both are deliberate:

1. **`(1 − FPt/Nt)` penalises the total *duration* of false positives.** Flagging 33% of the
   timeline destroys your precision even if you caught every event. This is why plain iForest
   scores 0.00007 on Mission 1 despite 0.585 recall.
2. **β = 0.5 weights precision 4× more than recall.** Mission control finds false alarms more
   disruptive than misses.

The optimal strategy is therefore: **flag almost nothing, but be right when you do.**
Telemanom-Pruned catches only 42% of events yet scores 0.786 — its precision is 0.999.

## Aspect 3 — alarming precision

```
alarming_precision = TP / (TP + redundant_detections)
```

`redundant_detections` counts, for each ground-truth event, how many *extra* predicted events
overlap it beyond the first. GDNv2 scored **0.228**, meaning ~4.4 fragmented alarms per real
event. Morphological closing fixed this to **0.7955** with no loss of F0.5.

## Aspect 4 — ADTQC

Scores detection *timing* against a curve that peaks when you detect exactly at onset and decays
for both early and late detection. Reports `Nb_Before`, `Nb_After`, `AfterRate`, and `Total`.

**This aspect is in direct tension with aspect 1a**, and ESA says so:

> *"pruning significantly decreases subsystem-/channel-aware, detection timing (ADTQC), and
> affiliation-based scores, so the anomalies may be harder to identify."*

Evidence from the published tables:

| Algorithm | Event-wise F0.5 | ADTQC |
|---|---|---|
| Telemanom-ESA-Pruned | **0.786** | 0.197 |
| DC-VAE-ESA STD3 | 0.003 | **0.901** |
| SMED (PHM 2025) | 0.801 | 0.160 |
| **Ours (GDNv4)** | **0.853** | 0.186 |

Everything that wins 1a loses 4, and vice versa. Beating all five simultaneously would itself be
a publishable result — and is not guaranteed to be possible.

---

## Our two scorers

### `federated/src/official_score.py` — byte-faithful replica of aspect 1a/3/5

Replicates `timeeval/core/experiments.py` exactly:
1. read `labels.csv`, strip timezones
2. keep events **fully inside** the scored range (`StartTime ≥ min`, `EndTime ≤ max`)
3. merge the **last four** `anomaly_types` columns by ID
4. filter to target channels **without deduplicating**
5. `ESAScores(betas=0.5, select_labels={"Category": ["Rare Event","Anomaly"]})`
   called as `metric.score(y_true.drop(columns=["Channel"]), test_data_scores)`

**Validation:** re-scoring the official `subsequence_if` container output gives
`EW_F_0.50 = 0.9487`, `precision = 0.9509`, `recall = 0.94` — **identical to the harness.**

### `models/gdnv4/src/esa_metrics.py` — all five aspects

Adds `ChannelAwareFScore` (with `subsystems_mapping` from `channels.csv`), `ADTQC`, and
per-channel `ESAScores`. Both take **per-channel** `y_pred` dicts, exactly as the harness builds them.

**Validation:** reproduces the known gap-detector score **0.41237** exactly.

`global_f05()` is a fast path computing only aspect 1a — used during hyper-parameter search
(1 metric instead of 14, ~14× faster; this is what made the 432-config search feasible).

---

## Ground truth we verified

| Check | Result |
|---|---|
| Windowed iForest, M2 lightweight, via real Docker + TimeEval | **0.9487** vs published **0.949** ✅ |
| iForest / HBOS / PCC, M1 lightweight | 0.00007 / 0.00014 / 0.00013 — reproduces "point detectors fail on M1" ✅ |
| Gap detector re-scored through `esa_metrics.py` | 0.41237, matches the direct scorer ✅ |

This is why we trust every downstream number.

---

## Rules we follow (and you should too)

1. **Never report oracle-threshold numbers** alongside published baselines. Only train-tuned or
   self-thresholding numbers are comparable.
2. **Tune on the train period only.** ESA-ADB ships label columns in the train CSVs and lists
   "learn from known anomalies" (R4) as *desired*, so this is legitimate — using test labels is not.
3. **Pre-register.** Print and freeze the winning config before the test evaluation runs.
4. **Report all five aspects.** Reporting only 1a hides real regressions (ours: ADTQC).
5. **Watch the comparison baseline.** SMED compares only against Telemanom-ESA, so their
   "Best-ESA" column for M2 reads 0.842 when the actual best is Windowed iForest at 0.949.
