#!/usr/bin/env bash
# =============================================================================
#  Build ESA-ADB's official baseline algorithm images (iForest, Windowed iForest,
#  HBOS, PCC, KNN) as local Docker images, so TimeEval can run them offline.
# =============================================================================
#  Why: the official images live in a private registry (pull = "access forbidden"),
#  and the stock python3-base image no longer builds (Debian buster apt repos are
#  archived). docker/python3-base/ is a patched copy that skips apt-get.
#
#  Usage:   bash preprocessing/official_baselines/build_official_images.sh /path/to/ESA-ADB
#  Builds natively for this machine (amd64 emulation on Apple Silicon is too slow).
# =============================================================================
set -euo pipefail
ESA_ADB="$(cd "${1:?pass the path to the ESA-ADB repo}" && pwd)"
ALG="$ESA_ADB/TimeEval-algorithms"                     # official algorithm Dockerfiles
PATCHED="$(cd "$(dirname "$0")/docker" && pwd)"        # our patched base image
R=registry.gitlab.hpi.de/akita/i                       # tag names TimeEval expects

echo "=== [1/3] base: python3-base (patched, no apt) ==="
docker build -t $R/python3-base:latest "$PATCHED/python3-base"
echo "=== [2/3] base: pyod-1.1.2 ==="
docker build -t $R/pyod-1.1.2:latest "$ALG/0-base-images/pyod-1.1.2"
echo "=== [3/3] algorithms ==="
for a in subsequence_if iforest hbos pcc knn; do
  echo "--- building $a ---"
  docker build -t $R/$a:latest "$ALG/$a"
done
docker images | grep akita
