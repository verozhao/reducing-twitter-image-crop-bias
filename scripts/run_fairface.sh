#!/usr/bin/env bash
# FairFace (Karkkainen and Joo, WACV 2021) as a second dataset: validation split, padding 1.25.
# Downloads the photos and labels from the links in the FairFace README, keeps people labeled
# Black or White and aged 20 or older, and runs the same pipeline in its own folder, work/fairface/
# (no DeepGaze). Aggregate tables and figures are copied to results/fairface/.
# Runtime on a 4-core Intel Mac: about 2 hours, most of it Twitter's program.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || { echo "Create the environment first (see README)"; exit 1; }
[ -x third_party/twitter/candidate_crops ] || ./scripts/get_twitter_model.sh
RAW=work/fairface_raw
mkdir -p "$RAW"
# Google Drive file ids from https://github.com/joojs/fairface
[ -f "$RAW/fairface_label_val.csv" ] || $PY -I -m gdown 1wOdja-ezstMEp81tX1a-EYkFebev4h7D -O "$RAW/fairface_label_val.csv"
[ -f "$RAW/fairface-img-margin125-trainval.zip" ] || $PY -I -m gdown 1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL -O "$RAW/fairface-img-margin125-trainval.zip"
[ -d "$RAW/images/val" ] || unzip -q -n "$RAW/fairface-img-margin125-trainval.zip" 'val/*' -d "$RAW/images"

export CROPBIAS_RUN_DIR=work/fairface CROPBIAS_IMAGES=work/fairface/images
export CROPBIAS_PAIRS_TEST=2000 CROPBIAS_PAIRS_VAL=500
for st in fairface faces dataset pairs composites; do $PY -m cropbias $st; done
$PY -m cropbias saliency --workers 8
$PY -m cropbias evaluate --workers 6
$PY -m cropbias analyze
$PY -m cropbias sources
mkdir -p results/fairface
cp -R work/fairface/results/. results/fairface/
