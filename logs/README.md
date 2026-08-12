# Run logs — the full audit trail

Kept deliberately. Every headline number in this repo traces back to one of these files, and the
iteration history is part of the honest disclosure: the *shape* of the detection pipelines was
chosen across these runs (final *parameters* are all train-tuned — see
[`../docs/04-RESULTS.md`](../docs/04-RESULTS.md) §8).

## Current models

| Log | What it is |
|---|---|
| `extract.log` | availability-feature extraction from the raw pickles |
| `gap_analysis.log` | ⭐ **the key finding** — count==0 → P(anomaly) 56.7%; gap detector alone 0.412 @ P=1.0 |
| `diagnose.log` | per-event signal table; the analysis that redirected the project |
| `M1_v1 … M1_v3.log` | GDNv2 Mission-1 iterations |
| `M1_FINAL.log` | ⭐ GDNv2 Mission-1 → **0.8197** |
| `M2_v1 … M2_v3.log` | GDNv2 Mission-2 iterations; `M2_v3` is the ⭐ **0.9644** winner |
| `miss_analysis.log` | which events are still missed and why → motivated block-local z |
| `M1_v4.log` | ⭐ GDNv4 Mission-1 → **0.8532** (432-config train grid, then one test eval) |
| `M2_v4.log` | GDNv4 on Mission 2 — **run in progress at time of commit** |

## `phase-a/` — earlier work, before the availability features

| Log | What it is |
|---|---|
| `official_M2_subif.log` | ⭐ **Windowed iForest via real Docker + TimeEval = 0.9487** vs published 0.949 — the validation that everything else rests on |
| `official_M1.log` | official iForest / HBOS / PCC on M1 → 0.00007 / 0.00014 / 0.00013 |
| `train_m1.log`, `train_m1_v2.log` | first single-mission GDN training runs |
| `tune_best.log`, `tune_telemanom.log` | threshold tuning; the Telemanom-pruning port (M1 0.080 → 0.337) |
| `federated.log`, `federated_official.log`, `federated_v2.log` | cross-mission federation + the personalisation ablation |
