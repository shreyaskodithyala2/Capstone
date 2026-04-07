import pandas as pd
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────
RAW_DIR = Path("ChannelsData")
OUTPUT_DIR = Path("ResampledData")

GLOBAL_START = pd.Timestamp("2000-01-01 00:00:00", tz="UTC")
GLOBAL_END = pd.Timestamp("2014-01-01 00:00:00", tz="UTC")
FREQ = "30s"

CHANNELS_TO_PROCESS = [
	"channel_1",
	"channel_2",
	"channel_3",
	"channel_4",
	"channel_5",
	"channel_6",
	"channel_7",
	"channel_8",
	"channel_9",
	"channel_10",
	"channel_11",
	"channel_12",
	"channel_13",
	"channel_17",
	"channel_18",
	"channel_19",
	"channel_20",
	"channel_25",
	"channel_26",
	"channel_27",
	"channel_28",
	"channel_34",
	"channel_35",
	"channel_36",
	"channel_37",
	"channel_64",
	"channel_65",
	"channel_66",
	"channel_67",
	"channel_68",
	"channel_69",
	"channel_74",
]


# ── Build master grid once ─────────────────────────────────────────────────
master_grid = pd.date_range(
	start=GLOBAL_START,
	end=GLOBAL_END,
	freq=FREQ,
	tz="UTC",
)


def find_channel_file(base_dir: Path, channel_name: str) -> Path:
	matches = list(base_dir.rglob(f"{channel_name}.csv"))
	if len(matches) == 0:
		raise FileNotFoundError(f"{channel_name}.csv not found under {base_dir}")
	if len(matches) > 1:
		raise ValueError(f"Multiple files found for {channel_name}: {matches}")
	return matches[0]


def _validate_channel_agg(agg: pd.DataFrame, channel_name: str) -> None:
	# Step 8: Final validation before saving
	if agg.isna().sum().sum() != 0:
		raise AssertionError(f"NaN found in {channel_name}")

	count_col = f"{channel_name}_count"
	std_col = f"{channel_name}_std"

	if not (agg[count_col] >= 0).all():
		raise AssertionError(f"Negative count in {channel_name}")
	if not pd.api.types.is_integer_dtype(agg[count_col].dtype):
		raise AssertionError(f"count dtype is not integer in {channel_name}: {agg[count_col].dtype}")
	if not (agg[std_col] >= 0).all():
		raise AssertionError(f"Negative std in {channel_name}")

	if len(agg) != len(master_grid):
		raise AssertionError(
			f"Row count mismatch: {len(agg)} vs {len(master_grid)} in {channel_name}"
		)
	if not (agg.index == master_grid).all():
		raise AssertionError(f"Index mismatch in {channel_name}")


def process_channel(channel_name: str) -> Path:
	# Step 1: Load raw data
	raw_path = find_channel_file(RAW_DIR, channel_name)
	df = pd.read_csv(raw_path)
	df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
	df = df.sort_values("datetime")
	df = df.set_index("datetime")
	df = df[~df.index.duplicated(keep="first")]

	if channel_name not in df.columns:
		raise KeyError(
			f"Expected column '{channel_name}' in {raw_path}, found columns: {list(df.columns)}"
		)

	# Step 2: Resample to 30s windows
	agg = df[channel_name].resample(FREQ).agg(
		mean="mean",
		max="max",
		min="min",
		std="std",
		count="count",
	)

	# Step 3: Reindex to master grid
	agg = agg.reindex(master_grid)

	# Step 4: Forward fill mean, max, min ONLY
	agg[["mean", "max", "min"]] = agg[["mean", "max", "min"]].ffill()

	# Step 5: Backfill Leading NaNs
	agg[["mean", "max", "min"]] = agg[["mean", "max", "min"]].bfill()

	# Step 6: Zero-Fill std and count — NEVER forward fill these
	agg["std"] = agg["std"].fillna(0.0)
	agg["count"] = agg["count"].fillna(0).astype(int)

	# Step 7: Rename Columns
	agg.columns = [
		f"{channel_name}_mean",
		f"{channel_name}_max",
		f"{channel_name}_min",
		f"{channel_name}_std",
		f"{channel_name}_count",
	]
	agg.index.name = "datetime"

	_validate_channel_agg(agg, channel_name)

	count_col = agg[f"{channel_name}_count"]
	print(
		f"{channel_name}: OK — {len(agg):,} rows, "
		f"count>0 in {(count_col > 0).sum():,} windows "
		f"({(count_col > 0).mean() * 100:.1f}%)"
	)

	# Step 9: Save Output (overwrite existing file at same rel path)
	out_path = find_channel_file(OUTPUT_DIR, channel_name)
	agg.to_csv(out_path)
	return out_path


def main() -> None:
	print(
		f"Master grid: {len(master_grid):,} slots "
		f"({GLOBAL_START} → {GLOBAL_END} @ {FREQ})"
	)
	print(f"\nProcessing {len(CHANNELS_TO_PROCESS)} channels...\n")

	for ch in CHANNELS_TO_PROCESS:
		print(f"Processing {ch}...")
		out_path = process_channel(ch)
		print(f"  Saved → {out_path}")

	print("\nDone. All channels processed and validated.")


if __name__ == "__main__":
	main()
