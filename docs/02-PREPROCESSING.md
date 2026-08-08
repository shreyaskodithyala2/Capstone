# 2 · Preprocessing — what happens to the data, and why

To *run* it, see [`preprocessing/README.md`](../preprocessing/README.md). This document explains
what the pipeline does and the reasoning behind each decision.

**Goal:** turn 76 irregular, independently-ticking streams into one clean rectangular table.

---

## Before and after, on real rows

### Raw `channel_41`, a 3-minute slice

```
                         channel_41
2009-01-23 19:24:24.942    0.804729
2009-01-23 19:24:54.942    0.801590
2009-01-23 19:25:24.942    0.804729
2009-01-23 19:25:54.942    0.803160
2009-01-23 19:26:24.942    0.952286
2009-01-23 19:26:27.030    0.969553
2009-01-23 19:26:47.442    0.818072
2009-01-23 19:26:54.942    0.803160
```

8 rows, irregular, millisecond offsets, one column, no labels.

### Preprocessed output, same window

```
          timestamp  ch_41     ch_42     ch_43     ch_44     ch_45     ch_46    is_an_41 … is_an_46
2009-01-23 19:24:00  0.803160  0.784738  0.777439  0.788945  0.807528  0.773740     0  0  0  0  0  0
2009-01-23 19:24:30  0.804729  0.784738  0.775064  0.792093  0.810920  0.769517     0  0  0  0  0  0
2009-01-23 19:25:00  0.801590  0.791163  0.775064  0.788945  0.813463  0.773740     0  0  0  0  0  0
2009-01-23 19:25:30  0.804729  0.783935  0.774274  0.785796  0.814311  0.770362     0  0  0  0  0  0
2009-01-23 19:26:00  0.803160  0.792770  0.774274  0.785796  0.814311  0.772051     0  0  0  0  0  0
2009-01-23 19:26:30  0.969553  0.933339  0.000000  0.958188  0.974563  0.928284     1  1  1  1  1  1
2009-01-23 19:27:00  0.803160  0.794377  0.774274  0.787371  0.816006  0.772051     0  0  0  0  0  0
```

*(13 of 175 columns shown.)*

**Look at row `19:26:30`.** Channel 43 collapses to **0.000000** while its five siblings jump *up*
to 0.93–0.97. Those six sensors are 97–100 % correlated — one going to zero while the others rise
is a **broken relationship**. A threshold detector sees nothing (0.0 is a legal value for that
channel). This is exactly the anomaly class a graph model is built to catch.

---

## The seven steps

### Step 1 — Choose a master grid

**Mission 1 → 30 s. Mission 2 → 18 s.** Chosen from the densest target channels. Every timestamp
becomes a clean multiple.

### Step 2 — Zero-Order Hold resampling

```python
param_df.reindex(resampled_range, method="ffill")
```

*At each grid point, take the last raw sample at or before that time.*

**The consequence people miss: within one bucket, only the LAST sample survives.** Verify on the
data above:

| Grid point | Raw samples ≤ that time | Survivor | Written |
|---|---|---|---|
| `19:26:30` | `…:24.942` (0.952286), `…:27.030` (0.969553) | the last | **0.969553** ✓ |
| `19:27:00` | `…:47.442` (0.818072), `…:54.942` (0.803160) | the last | **0.803160** ✓ |

Two real samples were permanently discarded. This is the lossy part — and it motivates Step 5.

**Why ZOH and not something else:**

| Alternative | Why rejected |
|---|---|
| Linear interpolation | **Invents data that was never measured**, and interpolates *across communication gaps* — manufacturing a smooth signal for periods when the satellite was silent |
| Mean over the window | **Smooths away spikes** — precisely what you're detecting |
| Backward fill | **Leaks the future** into the present |
| Leave NaN | Most models can't consume it; you lose the uniform matrix |

ZOH wins for three reasons: it is **physically what a spacecraft actually does** (hold the last
reading until a new one arrives); it **never invents data**; and it is **causal**.

### Step 3 — Derivative for counter channels (4–11)

```python
if 4 <= int(param.split("_")[1]) <= 11:
    param_df.value = np.diff(param_df.value, append=param_df.value[-1])
```

Channels 4–11 are **cumulative counters** — like an odometer, always increasing. Two problems if
fed raw: (a) the value grows without bound, so the model never sees late-mission values during
early training (**non-stationarity**), and (b) "odometer = 50,000" is meaningless; "it went up by
12" is the signal.

```
Before (cumulative):  1000, 1000, 1012, 1012, 1012, 1099
After  (derivative):     0,   12,    0,    0,   87,     0
```

### Step 4 — Stamp the labels

```python
param_df.loc[row["StartTime"]:row["EndTime"], "label"] = label_value
```

Using `AnnotationLabel`: `0` NOMINAL · `1` ANOMALY · `2` RARE_EVENT · `3` GAP · `4` INVALID.

**Why one integer column instead of three binary ones?** The categories are mutually exclusive at
a given instant, and keeping them distinct lets the official metric filter
(`select_labels={"Category": [...]}`) at evaluation time. Collapsing to binary during
preprocessing would destroy that.

### Step 5 — Anomaly-preserving correction ⭐

**The problem.** From Step 2, only the last sample in a bucket survives. So what happens when an
*annotated* sample isn't last?

We replicated the logic on real `channel_41` data and found actual cases:

```
30s bucket starting 2007-05-22 05:30:00  →  grid point 2007-05-22 05:30:30

                            value  label
2007-05-22 05:30:06.120  0.962489      1     ← ANOMALY
2007-05-22 05:30:25.032  0.806299      0     ← nominal, and it is LAST

  plain ZOH writes  : 0.806299, label 0   ← ANOMALY ERASED
  after restoration : 0.962489, label 1   ← PRESERVED
```

Without the fix, that row would claim "0.806299, perfectly normal" — while `labels.csv` insists
it's an anomaly. You'd train the model that anomalies look like nothing, *and* evaluate against a
label with no matching signal. Both corrupted.

**The implementation:**

```python
grouper = param_df.groupby(pd.Grouper(freq=resampling_rule))
for timestamp, group in grouper.indices.items():
    if len(group) <= 1:                                        # (a)
        continue
    org_elements = param_df.iloc[group]
    if org_elements.label.values[-1] != AnnotationLabel.NOMINAL.value:   # (b)
        continue
    is_annotated = (org_elements.label == ANOMALY) | (org_elements.label == RARE_EVENT)  # (c)
    if is_annotated.any():
        params_dict[param].loc[timestamp + resampling_rule] = org_elements[is_annotated].iloc[-1]
```

| Check | Purpose |
|---|---|
| **(a)** one sample → skip | ZOH keeps it faithfully; nothing can be lost |
| **(b)** last sample already annotated → skip | ZOH will pick it up anyway — avoids pointless writes |
| **(c)** only ANOMALY / RARE_EVENT | **Communication gaps are deliberately excluded** — you must not invent data for a period when the satellite was silent |

It restores the **whole row** (value *and* label) — restoring only the label would leave a label
with no signal, which is arguably worse.

This is the most thoughtful step in the official pipeline: a targeted repair that fires only
where damage occurs (a few thousand rows out of 7.4 M).

### Step 6 — Telecommands become 0→1→0 pulses

A telecommand fires at one instant. Plain ZOH would forward-fill `1` **forever** until the next
firing — the model would learn "telecommand permanently on".

`encode_telecommands()` plants a `0` immediately before and after each firing, producing an
isolated spike that survives resampling and decays immediately:

```
          timestamp  channel_41  telecommand_36
2007-04-17 17:34:30    0.841619       0.0
2007-04-17 17:35:00    0.838480       1.0   ← fired at 17:34:34.383
2007-04-17 17:35:30    0.838480       0.0
```

Only **priority ≥ 3** commands are kept (`telecommands_min_priority = 3`): **11 of 698** for M1,
**4 of 123** for M2. Priorities 0–1 (96 %) are routine housekeeping that doesn't change spacecraft
state — including them would add hundreds of near-empty noise columns.

### Step 7 — Assemble, split, register

Build one DataFrame on the common grid, order columns as
`[channels…, telecommands…, is_anomaly_*…]`, prepend `timestamp`, write CSV. Then
`DatasetAnalyzer` computes per-split statistics into `.metadata.json`, and `dm.add_dataset()`
registers a row in `datasets.csv`.

**Splits** — nested train windows sharing one test set:

| | Train cutoffs | Shared test |
|---|---|---|
| **M1** | 3, 10, 21, 42, 84 months | everything after **2007-01-01** (~7 yrs) |
| **M2** | 1, 5, 10, 21 months | everything after **2001-10-01** |

The nested design answers "how much training history does the model need?". This is
**semi-supervised**: train is mostly clean (measured **1.29 %** contamination for M1 lightweight),
so models learn *normal* and flag deviations.

---

## ⚠️ A discovered flaw in ZOH that matters enormously

This is a **finding from this project**, not something the paper reports.

We tested whether the number of raw samples arriving in each window correlates with anomalies:

```
channel_41, Mission 1 test period (7,364,160 grid rows, 1.806 % anomalous)

count = 0   →  P(anomaly) = 56.7 %    lift = 31×    ← 30.7 % of ALL anomalies are here
count = 1   →  P(anomaly) =  1.26 %   lift = 0.70×
count = 2   →  P(anomaly) =  1.12 %   lift = 0.62×
count = 10  →  P(anomaly) = 96.6 %    lift = 53×
```

Reproduced on `channel_43` (30.7 %) and `channel_45` (30.6 %).

**When no telemetry arrives in a 30 s window, there is a 56.7 % chance that window is annotated
anomalous — against a 1.8 % base rate.**

### Why this breaks forecasting models

Here is the actual preprocessed output during such a stretch:

```
          timestamp  channel_41  is_anomaly_channel_41
2007-04-14 20:37:00    0.456281            1
2007-04-14 20:37:30    0.456281            1
2007-04-14 20:38:00    0.456281            1
2007-04-14 20:38:30    0.456281            1
2007-04-14 20:39:00    0.456281            1
2007-04-14 20:39:30    0.456281            1
```

Every row labelled anomalous. Every value **identical** — because no data arrived, ZOH
forward-filled the same number.

A forecasting model sees a flat constant line, predicts "next = same as last", gets it exactly
right, and produces **near-zero error → no detection**. It is *structurally incapable* of finding
these. That is a precise, mechanical explanation for our Mission 1 recall of **0.09**.

The information isn't corrupted — it's **erased**. ZOH renders 30 % of Mission 1's anomalies as
the easiest possible signal to predict.

### And it explains the M1 / M2 asymmetry

| Mission | count=0 rows | P(anomaly \| count=0) | Share of anomalies | Our GDN score |
|---|---|---|---|---|
| **M1** (ch 43/45) | 71,914 | **56.7 %** | **30.7 %** | **0.337** ❌ |
| **M2** (ch 18/20/25) | 136 | 0.0 % | 0.0 % | **0.936** ✅ |

Mission 2 has essentially no count-zero phenomenon, and our model already performs near-SOTA
there. Mission 1 has it severely, and that is exactly where we're weak. Not a coincidence — a
mechanism.

### The fix (not yet implemented)

Add label-free **data-availability features**, computable from raw telemetry at inference time:

```
count       : raw samples in this window          (the core signal)
gap_flag    : 1 if count == 0                     (sharpest, binary)
time_since  : seconds since the last real sample  (captures gap DURATION — likely strongest)
```

This addresses **requirement R8** (irregular time series), unmet by every benchmark algorithm.

**Honest caveat to pre-empt in review:** a reviewer may ask whether this detects *missing data*
rather than *anomalies*. Our analysis **excluded** `Category == "Communication Gap"` — these are
genuine Anomaly and Rare Event annotations where telemetry also stopped. Telemetry loss is a
symptom of the anomalous state. Still, report scores **with and without** the feature.

---

## Should we have preprocessed differently?

We evaluated the Phase-2 custom pipeline (see `papers/phase2/`) against the official one.

| Method | Verdict | Reasoning |
|---|---|---|
| Interval analysis / channel profiling | ✅ **Keep** | Measurements verified correct (their "~79 readings per 30 s burst" — we measured 81). Good justification material |
| `count` feature | ✅ **Keep — best idea in it** | See above; ZOH structurally cannot represent it |
| mean/max/min/std aggregation | 🟡 **Ablation only, full channel set** | Big win during bursts (ZOH discards 98.8 %) but a no-op the other 99.2 % of the time — and channels 41–46 are classified ZOH under their own rule, so it never touches the benchmark subset |
| 30 s grid | ✅ **Keep** | Do **not** move to 90 s |
| 90 s grid | ❌ **Drop** | Channels 41–46 sample at 30 s; 90 s discards ⅔ of the resolution on the benchmark channels |
| `.resample().mean()` | ❌ **Drop** | Low-pass filter — smooths away the spikes you're detecting |
| Linear interpolation | ❌ **Drop** | Fabricates data across gaps up to 13 days long; contradicts the three-category label scheme |
| Priority {2,3} telecommands (30) | 🟡 Optional ablation | Official keeps ≥3 only (11). More context vs. comparability — test, don't assume |

**Overriding constraint:** the official pipeline is *verified correct* (Windowed iForest → 0.9487
vs published 0.949). Keep it as the reported pipeline; run variants as documented ablations.
