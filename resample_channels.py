"""
Resample all channel CSVs in ChannelsData/ to a uniform 30-second grid
using Zero-Order Hold (ZOH) and save into ResampledData/.
"""

import os
import time
import glob
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────
INPUT_ROOT = os.path.join(os.path.dirname(__file__), "ChannelsData")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "ResampledData")
RESAMPLE_FREQ = "30s"


def resample_channel(input_path: str, output_path: str) -> dict:
    """
    Resample a single channel CSV to a fixed 30-second grid with ZOH.

    Returns a dict with summary stats for logging.
    """
    # 1. Load and parse
    df = pd.read_csv(input_path, parse_dates=[0])

    # Handle malformed headers (e.g. channel_76.csv has header 'x'
    # but data rows have 2 comma-separated columns)
    if len(df.columns) < 2:
        # Derive channel name from filename
        basename = os.path.splitext(os.path.basename(input_path))[0]
        df = pd.read_csv(input_path, header=None, skiprows=1, names=["datetime", basename])
        df["datetime"] = pd.to_datetime(df["datetime"])

    datetime_col = df.columns[0]
    value_col = df.columns[1]

    # 2. Sort by datetime and set as index
    df = df.sort_values(datetime_col)
    df = df.set_index(datetime_col)

    original_rows = len(df)

    # 3-6. Resample to 30s grid
    #   .last()  → for sub-30s data, keeps the most recent value in each window
    #   .ffill() → forward-fill (ZOH) for intervals > 30s
    #   .bfill() → back-fill any leading NaNs from the first window
    df_resampled = (
        df[[value_col]]
        .resample(RESAMPLE_FREQ)
        .last()
        .ffill()
        .bfill()
    )

    resampled_rows = len(df_resampled)

    # 7. Reset index so datetime becomes a column again
    df_resampled = df_resampled.reset_index()
    df_resampled.columns = [datetime_col, value_col]

    # 8. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_resampled.to_csv(output_path, index=False)

    return {
        "original_rows": original_rows,
        "resampled_rows": resampled_rows,
        "has_nulls": int(df_resampled[value_col].isna().sum()),
    }


def main():
    csv_files = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", "*.csv"), recursive=True))

    if not csv_files:
        print(f"No CSV files found under {INPUT_ROOT}")
        return

    print(f"Found {len(csv_files)} channel CSV files to resample.\n")
    print(f"{'#':<4} {'Channel':<40} {'Original':>12} {'Resampled':>12} {'Nulls':>8} {'Time':>8}")
    print("-" * 90)

    total_start = time.time()

    for idx, input_path in enumerate(csv_files, 1):
        # Derive the relative path and build the output path
        rel_path = os.path.relpath(input_path, INPUT_ROOT)
        output_path = os.path.join(OUTPUT_ROOT, rel_path)

        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"{idx:<4} {rel_path:<40} {'SKIPPED (already exists)':>46}")
            continue

        t0 = time.time()
        stats = resample_channel(input_path, output_path)
        elapsed = time.time() - t0

        print(
            f"{idx:<4} {rel_path:<40} "
            f"{stats['original_rows']:>12,} "
            f"{stats['resampled_rows']:>12,} "
            f"{stats['has_nulls']:>8} "
            f"{elapsed:>7.1f}s"
        )

    total_elapsed = time.time() - total_start
    print("-" * 90)
    print(f"Done. Total time: {total_elapsed:.1f}s")
    print(f"Resampled files saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
