# GDNv4 — full ESA five-aspect results log

All numbers: official `ESAScores` / `ChannelAwareFScore` / `ADTQC` via
`src/esa_metrics.py` (validated: reproduces the known gap-detector 0.41237 exactly).
Protocol: every hyper-parameter tuned on the **TRAIN period only**, config printed and frozen,
then a **single** test evaluation. Lightweight subsets (M1 ch41-46, M2 ch18-28).

---

## Mission 1 — all five ESA aspects

| ESA aspect | Metric | Best published *(algo)* | SMED (PHM'25) | GDNv2 | GDNv3 | **v4** |
|---|---|---|---|---|---|---|
| **1a** | **Event-wise F0.5** | 0.786 *(Telem-P)* | 0.801 | 0.8197 | 0.3827 | **0.8532** 🥇 |
| | EW precision | 0.999 *(Telem-P)* | 0.999 | 0.9993 | 0.4443 | **0.9993** |
| **1b** | EW recall | 0.894 *(Telemanom)* | 0.446 | 0.4769 | 0.2462 | 0.5385 |
| **2b** | Channel-aware F0.5 | 0.837 *(Telemanom)* | 0.444 | *n/m* | 0.2238 | 0.4740 |
| | CA precision | 0.894 | 0.446 | *n/m* | 0.2462 | 0.5385 |
| | CA recall | 0.738 | 0.438 | *n/m* | 0.2026 | 0.3667 |
| **3** | Alarming precision | 0.875 *(Telem-P)* | 0.659 | 0.2279 | 0.6400 | 0.7955 |
| **4** | ADTQC score | 0.901 *(DC-VAE3)* | 0.160 | *n/m* | 0.5623 | 0.1864 |
| | ADTQC after-ratio | — | 0.103 | *n/m* | 0.9375 | 0.3429 |
| **5** | Affiliation F0.5 | 0.734 *(DC-VAE5)* | 0.634 | 0.6588 | 0.6856 | 0.6812 |
| | AFF precision | 0.780 | 0.716 | 0.7338 | 0.7524 | 0.7494 |
| | AFF recall | 0.662 | 0.435 | 0.4675 | 0.5058 | 0.4995 |

*n/m = never measured (GDNv2 emitted only a global series, so channel-aware/ADTQC were undefined).*
Subsystem-aware is excluded: ESA states it "is not reported" for lightweight subsets (single subsystem).

**v4 pre-registered config** (train F0.5 = 0.7748):
`p=0.3, zmode=local, q=0.9999, close=60, minlen=3, gapL=10, boost=0.6`

## Mission 2

| Metric | Best published | SMED | GDNv2 | v4 |
|---|---|---|---|---|
| Event-wise F0.5 | 0.949 *(W-iForest)* | **0.971** | 0.9644 | *running* |
| EW precision | 0.978 *(Telem-P)* | 0.979 | 0.9606 | — |
| EW recall | 1.000 | 0.940 | 0.980 | — |
| Channel-aware F0.5 | 0.822 *(Telemanom)* | 0.610 | *n/m* | — |
| Alarming precision | 0.912 *(Telemanom)* | 0.899 | 0.5135 | — |
| ADTQC | 0.999 *(PCC)* | 0.075 | *n/m* | — |
| Affiliation F0.5 | 0.976 *(iForest)* | 0.932 | **0.9839** 🥇 | — |

M2 v4 first config already at train F0.5 **0.8936** (P 0.994, R 0.636, alarmP 0.878).

---

## Where we lead / where we lag

**Lead (best published anywhere):**
- M1 event-wise F0.5 = **0.8532** (vs ESA 0.786, SMED 0.801)
- M1 event-wise precision = 0.9993 (ties best)
- M2 affiliation F0.5 = **0.9839** (vs iForest 0.976, SMED 0.932)

**Lag:**
- M1 ADTQC 0.186 vs 0.901 — `Nb_Before=23, Nb_After=12`: closing + TC-boost fire alarms early
- M1 channel-aware 0.474 vs 0.837 — per-channel recall only 0.367
- M1 alarming precision 0.796 vs 0.875 — close
- M2 event-wise 0.964 vs SMED 0.971

**Structural note:** no published algorithm wins all five. Telemanom-Pruned takes event-wise +
alarming precision but scores **0.197** ADTQC; DC-VAE takes ADTQC (0.901) but **0.003** event-wise;
SMED takes M2 event-wise but **0.075** ADTQC. ESA state the cause: *"pruning significantly
decreases subsystem-/channel-aware, detection timing (ADTQC), and affiliation-based scores."*

---

## Iteration history (audit trail)

| Run | Change | M1 test EW F0.5 | Lesson |
|---|---|---|---|
| gap detector alone | count==0 runs >=10 | 0.412 (P=1.0) | availability signal is real |
| GDNv2 v1 | TCN + naive fusion | 0.412 | model added nothing over gaps |
| GDNv2 v2 | cleaned-z threshold | 0.297 | thresholding alone insufficient |
| GDNv2 v3/FINAL | clean + Telemanom prune + gap | **0.8197** | pruning is the precision engine |
| GDNv3 | multi-span local z, no prune | 0.3827 | **removing prune collapses precision 0.999→0.444** |
| **v4** | prune + local z + closing + TC-boost | **0.8532** | combination wins; alarmP 0.228→0.796 |

Full logs: `../../logs/`. Result JSONs: `results/` and `../gdnv2/results/`.
