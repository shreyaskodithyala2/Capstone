"""
analyze_m1_intervals.py
-----------------------
Scope: ESA-Mission1 / ChannelsData / Group*/ channel_*.csv  (76 channels)

For each channel:
  - Parse the 'datetime' column (handles irregular/missing timestamps)
  - Compute consecutive time differences (seconds) using pandas diff()
  - Calculate: Mean, Median, Mode, Std Dev, Min, Max interval
  - Dominant ratio: % of intervals equal to the mode (rounded to 2 dp)
  - Top 3 most frequent interval values with their counts
  - First 5 and last 5 intervals for consistency checking
  - Skip files with < 3 valid rows

Output: one .txt file per channel under
        ESA-Mission1/interval_results/<GroupN>/<channel_N>.txt
"""

import os
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
MISSION1_DIR  = Path(__file__).resolve().parent          # .../ESA-Mission1
CHANNELS_ROOT = MISSION1_DIR / "ChannelsData"
OUTPUT_ROOT   = MISSION1_DIR / "interval_results"
MIN_ROWS      = 3   # need at least 3 rows to have 2 consecutive intervals

# ── Per-channel analysis ───────────────────────────────────────────────────────
def analyze_channel(csv_path: Path, output_path: Path) -> None:
    label = f"{csv_path.parent.name}/{csv_path.name}"

    # Read ONLY the datetime column (column index 0) — much faster on large files
    try:
        df = pd.read_csv(csv_path, usecols=[0], header=0, low_memory=False)
    except Exception as exc:
        _write(output_path, label, f"ERROR reading file: {exc}")
        return

    df.columns = ["datetime"]

    # Parse, coerce bad values → NaT
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    n_raw   = len(df)
    df      = df.dropna(subset=["datetime"])
    n_valid = len(df)
    n_dropped = n_raw - n_valid

    if n_valid < MIN_ROWS:
        _write(output_path, label,
               f"SKIPPED — only {n_valid} valid timestamp rows "
               f"(need >= {MIN_ROWS}). Rows with NaT: {n_dropped}.")
        return

    # Sort, compute differences (vectorised)
    df = df.sort_values("datetime").reset_index(drop=True)
    intervals = df["datetime"].diff().dt.total_seconds().dropna()

    n_intervals = len(intervals)

    # ── Core statistics (all vectorised) ────────────────────────────────────
    mean_val   = intervals.mean()
    median_val = intervals.median()
    std_val    = intervals.std()
    min_val    = intervals.min()
    max_val    = intervals.max()

    # Round to 2 dp once — used for mode, dominant ratio, and top-3
    rounded = intervals.round(2)

    mode_series = rounded.mode()
    mode_val    = float(mode_series.iloc[0]) if not mode_series.empty else float("nan")

    # Dominant ratio: % of intervals that equal the mode (after rounding)
    dominant_count = int((rounded == mode_val).sum())
    dominant_pct   = dominant_count / n_intervals * 100

    # Top-3 most frequent interval values (rounded to 2 dp)
    top3 = rounded.value_counts().head(3)          # Series: value → count
    top3_lines = [
        f"    #{rank+1}  {val:.2f} s  →  {cnt:,} occurrences  ({cnt/n_intervals*100:.2f}%)"
        for rank, (val, cnt) in enumerate(top3.items())
    ]

    # First / last 5 intervals
    first5 = intervals.iloc[:5].round(4).tolist()
    last5  = intervals.iloc[-5:].round(4).tolist()

    report = "\n".join([
        "=" * 68,
        f"  Channel       : {label}",
        "=" * 68,
        f"  Total rows    : {n_raw:,}",
        f"  Valid rows    : {n_valid:,}  |  Dropped (NaT): {n_dropped:,}",
        f"  Total intervals: {n_intervals:,}",
        "-" * 68,
        "  INTERVAL STATISTICS  (seconds between consecutive timestamps)",
        "-" * 68,
        f"  Mean              : {mean_val:.4f} s",
        f"  Median            : {median_val:.4f} s",
        f"  Mode              : {mode_val:.2f} s",
        f"  Std Dev           : {std_val:.4f} s",
        f"  Min               : {min_val:.4f} s",
        f"  Max               : {max_val:.4f} s",
        "-" * 68,
        f"  Dominant interval : {mode_val:.2f} s",
        f"  Dominant count    : {dominant_count:,}  ({dominant_pct:.2f}% of all intervals)",
        "-" * 68,
        "  TOP 3 MOST FREQUENT INTERVALS:",
        *top3_lines,
        "-" * 68,
        "  FIRST 5 INTERVALS (s) :",
        f"    {first5}",
        "",
        "  LAST 5 INTERVALS (s)  :",
        f"    {last5}",
        "=" * 68,
        "",
    ])

    _write(output_path, label, report, raw=True)


def _write(path: Path, label: str, content: str, raw: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw:
        path.write_text(content, encoding="utf-8")
    else:
        path.write_text(
            f"Channel : {label}\nStatus  : {content}\n",
            encoding="utf-8"
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not CHANNELS_ROOT.exists():
        print(f"[ERROR] ChannelsData folder not found: {CHANNELS_ROOT}")
        return

    # Collect all group folders, sorted numerically
    group_dirs = sorted(
        [d for d in CHANNELS_ROOT.iterdir() if d.is_dir()],
        key=lambda p: int("".join(filter(str.isdigit, p.name)) or "0"),
    )

    total_channels = 0
    total_groups   = 0

    print("=" * 68)
    print("  ESA-Mission1  |  ChannelsData Interval Analysis")
    print(f"  Output → {OUTPUT_ROOT}")
    print("=" * 68)

    for group_dir in group_dirs:
        csv_files = sorted(
            group_dir.glob("channel_*.csv"),
            key=lambda p: int("".join(filter(str.isdigit, p.stem)) or "0"),
        )
        if not csv_files:
            continue

        total_groups += 1
        print(f"\n  [{group_dir.name}]  ({len(csv_files)} channel(s))")

        for csv_path in csv_files:
            out_path = OUTPUT_ROOT / group_dir.name / csv_path.with_suffix(".txt").name
            print(f"    {csv_path.name} ... ", end="", flush=True)
            analyze_channel(csv_path, out_path)
            print("✓")
            total_channels += 1

    print(f"\n{'=' * 68}")
    print(f"  Done.  Groups: {total_groups}   Channels processed: {total_channels}")
    print(f"  Results saved to: {OUTPUT_ROOT}")
    print("=" * 68)


if __name__ == "__main__":
    main()
