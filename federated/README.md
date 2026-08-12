# Federated cross-mission GDN

The capstone's title claim: **two real spacecraft as federated clients**, not time-slices of one.

> **Status:** built, validated, and ablated — but on the **earlier, weaker GDN**, before the
> availability features existed. Re-running it on the GDNv4 encoder is priority #3 in
> [`../docs/06-ROADMAP.md`](../docs/06-ROADMAP.md).

---

## The problem federation has to solve here

The two missions are structurally different, which is exactly what makes this non-IID in a
meaningful way — and what makes naive FedAvg impossible:

| | Mission 1 | Mission 2 |
|---|---|---|
| Lightweight channels | 6 (`channel_41…46`) | 11 (`channel_18…28`) |
| Sampling grid | 30 s | 18 s |
| Span | 14 years | 3.5 years |
| Intra-mission correlation | **0.97 – 1.00** | **median 0.03** |
| Events | 200 (118 anomalies) | 644 (613 rare events) |
| Best detection policy | Telemanom pruning | absolute robust-z |

Averaging weights directly fails immediately — the embedding tables are `6×64` and `11×64`:

```
RuntimeError: stack expects each tensor to be equal size, but got [384,384] and [704,704]
```

---

## The split (FedPer-style)

| Component | Federated? | Why |
|---|---|---|
| Node embeddings | ❌ **local** | shapes differ; they encode *this specific spacecraft* |
| TCN encoder | ✅ shared | operates per node, dimension-independent |
| Attention vectors `a₁, a₂` | ✅ shared | `d`-dimensional, independent of N |
| Per-node output head `Linear(d→1)` | ✅ shared | **redesigned specifically for this** |

The head redesign is the enabling change. GDN's original `Linear(N·d → N)` scales with channel
count and can never be aggregated across clients with different N.

## One round

```
for each client (M1, M2):
    local train with FedProx, μ = 0.01      # keeps clients from drifting apart
    upload SHARED operators only            # embeddings never leave the client
aggregate with TRIMMED MEAN                 # robust to one client dominating
broadcast
for each client:
    per-mission FINE-TUNE                   # personalisation — NOT optional
```

---

## Result: personalisation is mandatory

| | Solo | Fed, no fine-tune | Fed + fine-tune |
|---|---|---|---|
| M1 | 0.1765 | 0.1370 ❌ | **0.1892** ✅ |
| M2 | 0.7739 | 0.7820 | **0.8050** ✅ |

Without the fine-tune step federation actively **hurts** Mission 1 — the shared operators drift
toward Mission 2, which has more data and more events. With it, both missions beat solo.

Independently corroborated: Atefrad & Karami (2025) report the same pattern (0.74 → 0.86 from
fine-tuning) on the same M2 subset.

> ⚠️ These are **oracle-threshold** numbers. They compare our own variants against each other and
> must **never** be quoted alongside published baselines.

### The privacy argument is real, not decorative

Only the shared operators cross the wire. Node embeddings — the component that encodes each
spacecraft's channel structure and correlations — stay local. Two agencies can co-train a detector
without exchanging telemetry or revealing their subsystem topology.

---

## What the two detection stacks imply

M1 scores 0.8532 with pruning and 0.137 without; M2 scores 0.9644 without pruning and 0.014 with
it. Same encoder, same errors, **opposite** optimal decision rules.

Standard personalised FL personalises the last few *layers*. Here the thing that most needs
personalising is the **post-hoc detection policy** — and it is worth ~0.8 F0.5, far more than any
weight-sharing choice. That is contribution **C3** in the roadmap.

---

## Files

| File | Purpose |
|---|---|
| `federated.py` | ⭐ FedProx local training, trimmed-mean aggregation, personalisation fine-tune |
| `gdn.py` | the earlier GDN (pre-TCN) these experiments used |
| `common.py` | data loading, splits, shared config — ⚠️ contains **absolute paths** |
| `official_score.py` | ⭐ byte-faithful TimeEval scoring replica (validated at 0.9487) |
| `telemanom_prune.py` | faithful port of Telemanom `find_epsilon` + `prune_anoms` |
| `train_m1.py`, `tune_best.py` | single-mission training and threshold tuning |
| `official_repro.py` | reproduces the official Docker baselines |
| `make_light_csvs.py` | trims 175-column CSVs to the target channels — result-identical, and the fix for Docker OOM (exit 137) |

`official_score.py` and `telemanom_prune.py` are reused by both winning models. They are the two
files here that are **not** superseded.
