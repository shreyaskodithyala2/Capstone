#!/bin/bash
# Build official ESA-ADB algorithm images locally for linux/amd64 (faithful to the paper;
# manylinux amd64 wheels exist for the pinned old versions). Run on Mac via emulation.
set -e
DOCKER=/Applications/Docker.app/Contents/Resources/bin/docker
ALG=/Users/Shreyas/Downloads/Capstone/ESA/ESA-ADB/TimeEval-algorithms
PATCHED=/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/docker_build
R=registry.gitlab.hpi.de/akita/i
P=linux/amd64

echo "=== [1/3] base: python3-base (patched, no apt) ==="
$DOCKER build --platform $P -t $R/python3-base:latest "$PATCHED/python3-base"
echo "=== [2/3] base: pyod-1.1.2 ==="
$DOCKER build --platform $P -t $R/pyod-1.1.2:latest "$ALG/0-base-images/pyod-1.1.2"
echo "=== [3/3] algorithms ==="
for a in subsequence_if iforest hbos pcc knn; do
  echo "--- building $a ---"
  $DOCKER build --platform $P -t $R/$a:latest "$ALG/$a"
done
echo "=== DONE. Local images: ==="
$DOCKER images | grep akita
