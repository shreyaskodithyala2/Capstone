# 4 · Results — every number, honestly

All scores are on the **lightweight subsets** (M1 = `channel_41…46`, 6 channels; M2 =
`channel_18…28`, 11 channels), computed with ESA's own metric classes on ESA's own train/test
split. Raw JSONs in `models/*/results/`, full run logs in [`../logs/`](../logs/).

**Scoring rule we follow:** all tuning on the **train period** only, config printed and frozen
before test evaluation. Oracle-threshold numbers are labelled as such and never compared to
published baselines.

---

## Headline

| | **Ours** | Published best | Δ |
|---|---|---|---|
| **Mission 1** event-wise F0.5 | **0.8532** *(GDNv4)* | 0.786 *Telemanom-ESA-Pruned* | **+0.067** ✅ |
| **Mission 2** event-wise F0.5 | **0.9644** *(GDNv2)* | 0.949 *Windowed iForest* | **+0.015** ✅ |
| **Mission 2** affiliation F0.5 | **0.9839** *(GDNv2)* | 0.976 *iForest* | **+0.008** ✅ |

Mission 1 also beats **SMED** (Brancato et al., PHM Society 2025, 0.801) — as far as we can find,
**0.8532 is the best published Mission-1 result anywhere.**

---

## 1 · Pipeline validation — the result everything else rests on

| | Ours | Published | |
|---|---|---|---|
| **M2 Windowed iForest**, real Docker + TimeEval | **0.9487** (P 0.9509, R 0.94) | **0.949** | ✅ matches |
| M1 iForest / HBOS / PCC | 0.00007 / 0.00014 / 0.00013 | "≈0" | ✅ reproduces the failure |
| Gap detector via `esa_metrics.py` | 0.41237 | — | ✅ matches our direct scorer |

Our scorer is a byte-faithful replica of `timeeval/core/experiments.py`, so every number below is
comparable to the paper without running Docker.

---

## 2 · Mission 1 — all five ESA aspects

| # | Aspect | Best published *(algo)* | SMED 2025 | GDNv2 | GDNv3 *(ablation)* | **GDNv4** |
|---|---|---|---|---|---|---|
| **1a** | Event-wise F0.5 | 0.786 *(Telem-P)* | 0.801 | 0.8197 | 0.3827 | **0.8532** 🥇 |
| | EW precision | 0.999 | 0.999 | 0.9993 | 0.4443 | **0.9993** |
| **1b** | EW recall | 0.894 *(Telemanom)* | 0.446 | 0.4769 | 0.2462 | 0.5385 |
| **2b** | Channel-aware F0.5 | 0.837 *(Telemanom)* | 0.444 | *n/m* | 0.2238 | 0.4740 |
| **3** | Alarming precision | 0.875 *(Telem-P)* | 0.659 | 0.2279 | 0.6400 | 0.7955 |
| **4** | ADTQC total | 0.901 *(DC-VAE3)* | 0.160 | *n/m* | **0.5623** | 0.1864 ❌ |
| | ADTQC after-rate | — | 0.103 | *n/m* | 0.9375 | 0.3429 |
| **5** | Affiliation F0.5 | 0.734 *(DC-VAE5)* | 0.634 | 0.6588 | 0.6856 | 0.6812 |

*n/m = not measured (GDNv2 predates the full scoreboard). Aspect 2a (subsystem-aware) is not
reported for lightweight subsets — ESA states it is degenerate there.*

**Won:** 1a (the headline aspect), and EW precision.
**Lost:** 1b recall, 2b channel-aware, 3 alarming precision, 4 ADTQC, 5 affiliation.

### The ADTQC regression — diagnosed, not mysterious

`Nb_Before = 23, Nb_After = 12`. Two-thirds of detections fire *before* the annotated onset,
because `close = 60` and the telecommand boost extend detections backwards, and ADTQC penalises
early detection hard.

This is exactly the trade-off ESA documents:

> *"pruning significantly decreases subsystem-/channel-aware, detection timing (ADTQC), and
> affiliation-based scores, so the anomalies may be harder to identify."*

Telemanom-Pruned — the algorithm ESA calls best — scores **0.197** on ADTQC. We score 0.186 with a
*better* after-rate than SMED (0.343 vs 0.103). GDNv3 shows the other end of the trade-off
(ADTQC 0.5623, but F0.5 collapses to 0.3827).

**Fix identified, not yet run:** emit the *first index* of each merged run rather than the whole
closed block. Should cut `Nb_Before` at no cost to F0.5.

---

## 3 · Mission 2

| Aspect | Best published *(algo)* | SMED 2025 | **GDNv2** |
|---|---|---|---|
| Event-wise F0.5 | 0.949 *(Windowed iForest)* | **0.971** | 0.9644 |
| EW precision | 0.951 | — | 0.9606 |
| EW recall | 0.940 | — | 0.9800 |
| Alarming precision | — | — | 0.5135 |
| Affiliation F0.5 | 0.976 *(iForest)* | — | **0.9839** 🥇 |

We beat the ESA-published best (0.949) but **SMED beats us** (0.971). Note that SMED compares only
against Telemanom-ESA, so its own paper lists "best ESA" for M2 as 0.842 — the actual published
best is Windowed iForest at 0.949. See [`07-RELATED-WORK.md`](07-RELATED-WORK.md).

**The v4 stack on Mission 2 is still running** (432-config grid, ~200 done, best train F0.5 0.9048
so far). GDNv2's 0.9644 remains the reported number.

---

## 4 · Why the two missions need opposite detection stacks

Same encoder, same errors, two detection policies:

| Detection stack | Mission 1 | Mission 2 |
|---|---|---|
| Telemanom dynamic threshold + pruning | **0.8532** ✅ | **0.014** ❌ |
| Absolute robust-z threshold | 0.137 ❌ | **0.9644** ✅ |

M1's anomalies are sharp and rare inside a noisy background (gap-recovery spikes reach z ≈ 1149),
so pruning's "keep only what stands decisively above the crowd" is exactly right. M2's rare events
are subtle and numerous (613 of 644 events) — pruning throws nearly all of them away.

**This is a finding, not a nuisance.** It says a cross-mission federated system must personalise
the *decision layer*, not only the weights.

---

## 5 · Ablations

| Ablation | Result | What it proves |
|---|---|---|
| Remove Telemanom pruning (`gdnv4/ablations/run_v3.py`) | M1 F0.5 **0.3827**, precision 0.999 → 0.444 | Pruning is the precision engine |
| Gap detector alone | M1 **0.412 at precision 1.000** | Availability is a standalone detector |
| Global signal vs per-channel | channel-aware 0.123 → **0.474** | Per-channel emission is mandatory for aspect 2b |
| Quantile vs absolute z transfer (M2) | 0.9260 vs **0.9644** | Absolute transfer generalises better here |
| Closing off vs on | alarming precision 0.228 → **0.7955** | Fragmentation, not detection, was the problem |
| Block-local vs global z | missed events sit below global p99.9; 3.6 → 26.8 in isolation | Local normalisation lifts recall |

---

## 6 · Federation (earlier model, oracle threshold — internal comparison only)

| | Solo | Fed, no fine-tune | Fed + fine-tune |
|---|---|---|---|
| M1 | 0.1765 | 0.1370 ❌ | **0.1892** ✅ |
| M2 | 0.7739 | 0.7820 | **0.8050** ✅ |

Federation **plus personalisation** helps both missions; federation **without** personalisation
hurts M1. Consistent with Atefrad & Karami (2025), who see 0.74 → 0.86 from fine-tuning.

> ⚠️ These are oracle-threshold numbers, used only to compare our own variants against each other.
> They are **not** comparable to published baselines and must never be quoted as such.

---

## 7 · Earlier results, for the record

| Model | M1 | M2 | Scoring |
|---|---|---|---|
| iForest / HBOS / PCC (official Docker) | 0.00007 / 0.00014 / 0.00013 | — | official harness |
| Windowed iForest (official Docker) | <0.001 *(pub.)* | 0.9487 | official harness |
| GDN solo (Phase-2 style) | 0.080 | 0.923 | val-tuned |
| GDN federated + personalised | 0.018 | 0.936 | val-tuned |
| GDN + Telemanom pruning | 0.337 (P 1.00, R 0.09) | 0.014 | self-thresholding |
| **GDNv2** | **0.8197** | **0.9644** | train-tuned |
| **GDNv4** | **0.8532** | *(running)* | train-tuned |

The jump from 0.337 → 0.8197 on M1 is the availability features. The jump 0.8197 → 0.8532 is the
v4 detection stack.

---

## 8 · Limitations — read before quoting any number

1. **Single seed.** Every headline number is seed 0. M2's margin (+0.015) is roughly 1–2 events.
   A 3-seed mean ± std is mandatory before publication.
2. **ADTQC regression on M1** (0.186). Real, diagnosed, fix identified, not yet run.
3. **Design informed by test-outcome iterations.** Final *parameters* are all train-tuned and M2's
   rule was genuinely pre-registered, but the *shape* of the pipeline was chosen across logged
   iterations that saw test scores. Disclose this in any paper.
4. **The availability features come from raw timestamps the official CSVs erase**, so published
   baselines never had them. That asymmetry *is* the R8 contribution — but the
   Telemanom+availability ablation must be run to prove it is the feature, not the model.
5. **Scored via our validated replica**, not the full Docker harness end-to-end.
6. **Solo models.** Federated GDNv4 has not been run.
7. **Lightweight subsets only.** The full 76/100-channel sets are untouched.
8. **`Communication Gap` events are excluded** (per ESA), but remaining Anomaly/Rare-Event
   annotations also include telemetry stoppages. Report with and without the availability feature.
