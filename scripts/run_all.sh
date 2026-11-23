#!/usr/bin/env bash
# Full pipeline, from Wikidata to tables and figures. Every stage resumes where it stopped.
# Runtime on a 4-core Intel Mac: about 5 hours, most of it Twitter's program and DeepGaze.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || { echo "Create the environment first (see README)"; exit 1; }
[ -x third_party/twitter/candidate_crops ] || ./scripts/get_twitter_model.sh

$PY -m cropbias people
$PY -m cropbias download --small-originals
$PY -m cropbias faces
$PY -m cropbias dataset
$PY -m cropbias pairs
$PY -m cropbias composites
$PY -m cropbias saliency --workers 8
$PY -m cropbias deepgaze
$PY -m cropbias evaluate --workers 6
$PY -m cropbias analyze
