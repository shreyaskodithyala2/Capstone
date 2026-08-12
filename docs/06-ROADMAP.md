# 6 · Roadmap — done, remaining, and the research plan

---

## Where the capstone stands against its own scope

| Original Phase-1/2 objective | Status |
|---|---|
| Acquire and preprocess real multi-satellite telemetry | ✅ both missions, official pipeline, verified |
| Reproduce official baselines | ✅ Windowed iForest 0.9487 vs published 0.949 |
| Build a graph-based anomaly detector | ✅ GDNv2 / GDNv4 |
| Beat the published benchmark | ✅ M1 0.8532 vs 0.786 · M2 0.9644 vs 0.949 |
| Federated architecture across satellites | ⚠️ built and validated **on the earlier weak model**; not yet on GDNv4 |
| Full evaluation suite | ✅ all five ESA aspects implemented and reported |
| Deep baselines (Telemanom-ESA, DC-VAE) run locally | ❌ blocked on Mac — needs GPU/cloud |
| Full 76 / 100-channel sets | ❌ untouched |
| Seed stability | ❌ single seed |

**Honest summary:** the modelling objective is exceeded; the *federated* objective — the actual
title of the capstone — is validated in principle but not yet run on the winning model. That gap
is the top priority below.

---

## Immediate queue (before any paper submission)

### 1 · Seed stability — ~1 h, **mandatory**
Rerun GDNv2 M1 + M2 and GDNv4 M1 with seeds {0, 1, 2}; report mean ± std. M2's margin (+0.015) is
1–2 events; a single-seed claim will not survive review.

### 2 · ADTQC fix — ~1 h
Emit the **first index** of each merged run instead of the whole closed block. `Nb_Before = 23`
should drop sharply at no cost to F0.5. Currently the only aspect where we regressed against our
own earlier variant (0.186 vs GDNv3's 0.562).

### 3 · Federated GDNv4 — ~half day, **the capstone's thesis**
Embeddings local; TCN + attention + head shared; FedProx μ = 0.01, trimmed-mean aggregation,
per-mission fine-tune. Everything is already validated on the earlier model — this is a rerun on
the winning encoder.

### 4 · The killer ablation — ~half day
Telemanom-style LSTM **with** the availability features. Either outcome is publishable:
- If it also jumps → the *feature* is the contribution (R8), model-agnostic. Stronger claim.
- If it doesn't → the graph architecture is doing real work.

Either way it pre-empts the obvious reviewer critique: *"your baselines never had those inputs."*

### 5 · Full-harness confirmation — ~1 day
Wrap GDNv4 as a TimeEval algorithm and produce headline numbers through the actual Docker
pipeline rather than our validated replica.

### 6 · Chase SMED on Mission 2
They score 0.971 vs our 0.9644. The M2 v4 grid search is running now; a top-k sweep is the other
obvious lever (M2's channels correlate at median 0.03, and we currently run a complete graph,
which disables graph learning exactly where it would help most).

---

## The research plan — what makes this novel

Three contributions, in descending confidence:

### C1 · Availability-aware detection (R8) — *strongest, already proven*

ESA-ADB's own preprocessing erases ~30% of Mission 1's anomalies via Zero-Order-Hold. Recovering
label-free availability features fixes it, and the gap detector alone scores **0.412 at precision
1.000**. This is a finding about **the benchmark**, not just about our model — which makes it
useful to everyone working on it.

*Needed to bulletproof:* the killer ablation (#4).

### C2 · Genuine cross-mission federation — *the capstone's title claim*

Every prior FL-on-telemetry paper federates **time-slices of one spacecraft**. We federate **two
real, structurally different spacecraft**: different sampling grids (30 s vs 18 s), different
channel counts (6 vs 11), different intra-mission correlation (0.97–1.00 vs median 0.03), and —
the key point — **different optimal detection policies**.

The architectural answer is the FedPer-style split plus mandatory personalisation, already
validated: federation **without** fine-tuning *hurts* M1 (0.1765 → 0.1370); **with** it, both
missions improve.

*Needed:* run it on GDNv4 (#3).

### C3 · Personalised detection policies under federation — *the novel extension*

The sharpest empirical fact in this project:

| Detection stack | Mission 1 | Mission 2 |
|---|---|---|
| Telemanom prune | **0.8532** | **0.014** |
| Absolute robust-z | 0.137 | **0.9644** |

The *same* encoder, the *same* errors, and the missions need **opposite** decision rules. Standard
personalised FL personalises the last few *layers*. Nobody, as far as we can find, personalises
the **post-hoc detection policy** while federating the representation — and here that choice is
worth 0.8 F0.5, far more than any weight-sharing scheme.

**Framing:** *"Federated representation learning with mission-personalised detection policies for
cross-satellite anomaly detection."* C1 supplies the input features, C2 the architecture, C3 the
novelty.

---

## Later work

- **Top-k sweep** — `topk = N−1` (a complete graph) currently disables GDN's graph learning.
  Defensible on M1 (6 channels, correlation 0.97–1.00), a real loss on M2 (median 0.03).
- **STGLR directed-graph ablation** — two embedding tables + antisymmetric combination gives
  directed edges. Cheap, and would make the learned telecommand→channel edges even more
  interpretable.
- **Per-window dynamic graph** — GDN and STGLR both freeze the graph at inference. A genuinely
  per-window graph is unclaimed territory.
- **Full 76 / 100-channel sets** — Kaggle free GPU.
- **A third federated client.** OPS-SAT is incompatible (univariate fragments); Mission 3 is
  off-limits (breaks comparability). SMAP/MSL work as an extra client but not as a headline
  benchmark.

---

## Alternative pivots, if the federated angle stalls

| Pivot | Pitch | Risk |
|---|---|---|
| **Benchmark-integrity paper** | "ESA-ADB's preprocessing erases 30% of Mission 1's anomalies" — a short, high-impact dataset paper. C1 alone carries it. | Narrower contribution, but the evidence is already complete |
| **Interpretability** | The learned telecommand→channel edges recover a command-to-sensor mapping ESA deliberately anonymised. Operationally valuable. | Needs validation against something; ESA anonymised the ground truth |
| **Timing-first detection** | Everyone trades ADTQC for precision. A method that wins *both* would be genuinely new. | Hard — DC-VAE gets 0.901 ADTQC with 0.003 F0.5; the frontier is real |
| **Efficiency** | 38k params, laptop-trained, beats the SOTA. Frame as edge/on-board deployment for constellations. | Systems venue, not ML; needs latency/power numbers |

---

## Compute

GDNv2/v4 run comfortably on an M4 MacBook Air (MPS): ~40 s/epoch, full-series error passes ~53 s,
~2.7 GB RAM, ~15 min end-to-end per mission. The 432-config M1 grid takes ~45 min; the M2 grid is
much slower (~4 h+) because M2 has ~350 in-range events to score per config.

Full channel sets and the TensorFlow-GPU baselines (Telemanom-ESA, DC-VAE) need Kaggle's free GPU
— they are blocked on Mac.
