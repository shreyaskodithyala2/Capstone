#!/usr/bin/env bash
# =============================================================================
#  Run the OFFICIAL ESA-ADB preprocessing, end to end.
# =============================================================================
#  Usage:
#    bash preprocessing/run_preprocessing.sh <ESA-ADB repo> <Mission1 dir> <Mission2 dir>
#
#  Example:
#    bash preprocessing/run_preprocessing.sh \
#         ~/Downloads/Capstone/ESA/ESA-ADB \
#         ~/Downloads/Capstone/ESA/ESA-Mission1 \
#         ~/Downloads/Capstone/ESA/ESA-Mission2
#
#  Runtime: ~2h for Mission1, ~2h for Mission2 on an M4 MacBook Air.
#  Disk:    ~34 GB of output. Make sure you have it free.
#
#  SAFE TO RE-RUN: every write is guarded by `if not target_filepath.exists()`,
#  so if it is interrupted it resumes from where it stopped.
# =============================================================================
set -euo pipefail

ESA_ADB="${1:?pass ESA-ADB repo path}"
M1="${2:?pass Mission1 dir}"
M2="${3:?pass Mission2 dir}"
ESA_ADB="$(cd "$ESA_ADB" && pwd)"; M1="$(cd "$M1" && pwd)"; M2="$(cd "$M2" && pwd)"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate timeeval

LOGS="$ESA_ADB/../preprocessing_logs"; mkdir -p "$LOGS"

# --- sanity: the raw data must look like the Zenodo layout -------------------
for D in "$M1" "$M2"; do
  for REQ in channels telecommands labels.csv channels.csv telecommands.csv anomaly_types.csv; do
    [[ -e "$D/$REQ" ]] || { echo "ERROR: $D/$REQ missing - is this really an ESA-AD mission folder?"; exit 1; }
  done
done
echo "==> raw data layout OK"
echo "    Mission1: $(ls "$M1/channels" | wc -l | tr -d ' ') channel files"
echo "    Mission2: $(ls "$M2/channels" | wc -l | tr -d ' ') channel files"

# --- 1. back up anomaly_types.csv (step 3 OVERWRITES it in place) ------------
for D in "$M1" "$M2"; do
  if [[ ! -f "$D/anomaly_types.orig.csv" ]]; then
    cp "$D/anomaly_types.csv" "$D/anomaly_types.orig.csv"
    echo "==> backed up $(basename "$D")/anomaly_types.csv -> anomaly_types.orig.csv"
  fi
done

# --- 2. Mission 1 -------------------------------------------------------------
echo ""
echo "==> [1/4] Mission1 preprocessing (long; ~2h). Log: $LOGS/m1_prep.log"
cd "$ESA_ADB/notebooks/data-prep"
python -u Mission1_semisupervised_prep_from_raw.py "$M1" 2>&1 | tee "$LOGS/m1_prep.log" | grep -E "written dataset|analyzed dataset|processed source" || true

# --- 3. Mission 2 -------------------------------------------------------------
echo ""
echo "==> [2/4] Mission2 preprocessing (long; ~2h). Log: $LOGS/m2_prep.log"
python -u Mission2_semiunsupervised_prep_from_raw.py "$M2" 2>&1 | tee "$LOGS/m2_prep.log" | grep -E "written dataset|analyzed dataset|processed source" || true

# --- 4. regenerate anomaly type metadata --------------------------------------
echo ""
echo "==> [3/4] infer_anomaly_types for Mission1"
cd "$ESA_ADB/scripts"
python -u infer_anomaly_types.py -i "$M1" -o "$M1" > "$LOGS/infer_m1.log" 2>&1
echo "==> [4/4] infer_anomaly_types for Mission2"
python -u infer_anomaly_types.py -i "$M2" -o "$M2" > "$LOGS/infer_m2.log" 2>&1

# --- verify -------------------------------------------------------------------
echo ""
echo "==> verifying output"
python - "$ESA_ADB" <<'PY'
import sys, os, glob, pandas as pd
base = os.path.join(sys.argv[1], "data", "preprocessed")
spec = {
 "ESA-Mission1-semi-supervised": (["3_months","10_months","21_months","42_months","84_months"], "84_months", 30, 76, 11),
 "ESA-Mission2-semi-supervised": (["1_months","5_months","10_months","21_months"],              "21_months", 18, 100, 4),
}
ok = True
for coll,(splits,test,grid,nch,ntc) in spec.items():
    d = os.path.join(base,"multivariate",coll)
    csvs = glob.glob(d+"/*.csv"); metas = glob.glob(d+"/*.metadata.json")
    print(f"\n{coll}: {len(csvs)} CSVs (expect {len(splits)+1}), {len(metas)} metadata (expect {len(splits)})")
    ok &= len(csvs)==len(splits)+1 and len(metas)==len(splits)
    for s in splits:
        h = pd.read_csv(f"{d}/{s}.train.csv", nrows=2); c=list(h.columns)
        got_ch = sum(x.startswith("channel_") for x in c)
        got_tc = sum(x.startswith("telecommand_") for x in c)
        got_lb = sum(x.startswith("is_anomaly_") for x in c)
        dt = (pd.to_datetime(h.timestamp).iloc[1]-pd.to_datetime(h.timestamp).iloc[0]).total_seconds()
        good = c[0]=="timestamp" and got_ch==nch and got_tc==ntc and got_lb==nch+ntc and dt==grid
        ok &= good
        print(f"  {s}.train: cols={len(c)} ch={got_ch} tc={got_tc} lab={got_lb} grid={dt:.0f}s  {'OK' if good else 'FAIL'}")
dd = pd.read_csv(os.path.join(base,"datasets.csv"))
for c in ["ESA-Mission1","ESA-Mission2"]:
    print(f"datasets.csv {c}: {len(dd[dd.collection_name==c])} rows")
print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
PY

echo ""
echo "==> preprocessing complete. Output in: $ESA_ADB/data/preprocessed/multivariate/"
