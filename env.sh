# Optional: `source env.sh` before running anything in Main/.
# src/config.py already derives every path from the folder layout, so this is only needed to
# point the code somewhere else (e.g. data on another disk); override any of the variables below.
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate timeeval

export ESA_MAIN="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export ESA_ROOT="$(dirname "$ESA_MAIN")"          # holds ESA-ADB/, ESA-Mission1/, ESA-Mission2/
export ESA_DATA="$ESA_MAIN/data"                  # *_feats.npz (+ data/full/ arrays)
export ESA_OUT="$ESA_MAIN/results"                # checkpoints + result JSONs
export TIMEEVAL_PATH="$ESA_ROOT/ESA-ADB"          # ESA's official metric package
export PYTHONPATH="$ESA_MAIN/src:$PYTHONPATH"
echo "ESA_ROOT=$ESA_ROOT"
