#!/usr/bin/env bash
# =============================================================================
#  ESA-ADB environment setup
# =============================================================================
#  Creates the `timeeval` conda environment and installs everything needed to
#  (a) run the official preprocessing, (b) score with the official metric, and
#  (c) train the GDN models.
#
#  Usage:   bash preprocessing/setup_env.sh /path/to/ESA-ADB
#  Example: bash preprocessing/setup_env.sh ~/Downloads/Capstone/ESA/ESA-ADB
#
#  Tested on macOS arm64 (M4) and should work on Linux x86_64.
# =============================================================================
set -euo pipefail

ESA_ADB="${1:-}"
if [[ -z "$ESA_ADB" || ! -f "$ESA_ADB/setup.py" ]]; then
  echo "ERROR: pass the path to your cloned ESA-ADB repo."
  echo "   git clone https://github.com/kplabs-pl/ESA-ADB"
  echo "   bash preprocessing/setup_env.sh /path/to/ESA-ADB"
  exit 1
fi
ESA_ADB="$(cd "$ESA_ADB" && pwd)"
echo "==> ESA-ADB repo: $ESA_ADB"

# --- locate conda -------------------------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" ]]; then
  echo "ERROR: conda not found. Install miniforge/miniconda first."; exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- 1. create the env --------------------------------------------------------
if conda env list | grep -qE "^timeeval\s"; then
  echo "==> conda env 'timeeval' already exists, reusing it"
else
  echo "==> creating conda env 'timeeval' from environment.yml"
  conda env create -f "$ESA_ADB/environment.yml" || {
    echo "==> environment.yml failed; falling back to a manual pinned env"
    conda create -y -n timeeval python=3.9 \
      "numpy=1.21.6" "pandas=1.5.3" "scipy" "scikit-learn" "statsmodels" "matplotlib" "tqdm" "jinja2" "psutil"
  }
fi
conda activate timeeval
echo "==> python: $(python --version)"

# --- 2. the local (modified) TimeEval package --------------------------------
# NOTE --no-deps is REQUIRED: ESA-ADB's setup.py contains a malformed pin
# (`requests====2.31.0`, four equals signs) that breaks pip's resolver.
# --no-build-isolation avoids a PEP-660 editable-build hook failure.
echo "==> installing local TimeEval (editable, no-deps)"
pip install -e "$ESA_ADB" --no-deps --no-build-isolation -q

# --- 3. runtime deps the repo forgets to declare ------------------------------
echo "==> installing missing runtime deps"
pip install -q natsort more-itertools durations numpyencoder portion prts

# --- 4. docker SDK (only needed to run the official baselines) ----------------
# The pinned docker==4.4.* is incompatible with modern urllib3 (>=2.0):
#   TypeError: request() got an unexpected keyword argument 'chunked'
# 7.1.0 fixes it and is API-compatible with everything TimeEval calls.
echo "==> installing docker SDK 7.1.0"
pip install -q -U "docker==7.1.0"

# --- 5. PyTorch for the GDN work ----------------------------------------------
# --no-deps protects numpy 1.21.6 (torch would otherwise pull numpy>=2 and
# silently break the pandas/scipy/sklearn ABI). We then add torch's small
# pure-python deps by hand.
echo "==> installing PyTorch (protecting numpy 1.21.6)"
pip install -q --no-deps torch
pip install -q --no-deps typing_extensions sympy networkx filelock mpmath fsspec

# --- 6. verify ----------------------------------------------------------------
echo ""
echo "==> verification"
python - <<'PY'
import numpy, pandas
print(f"   numpy  {numpy.__version__}   (MUST be 1.21.x)")
print(f"   pandas {pandas.__version__}")
from timeeval import DatasetManager, Datasets
from timeeval.datasets import DatasetAnalyzer, DatasetRecord, AnomalyLength
from timeeval.metrics import ESAScores
print("   timeeval + ESAScores  OK")
import natsort, more_itertools
print("   natsort, more_itertools  OK")
try:
    import torch
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   torch {torch.__version__}  device={dev}")
except Exception as e:
    print(f"   torch NOT available: {e}")
assert numpy.__version__.startswith("1.21"), "numpy was upgraded - ABI will break!"
print("\n   ALL CHECKS PASSED")
PY

echo ""
echo "==> done. Activate with:"
echo "    source \"$CONDA_BASE/etc/profile.d/conda.sh\" && conda activate timeeval"
