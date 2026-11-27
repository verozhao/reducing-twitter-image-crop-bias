#!/usr/bin/env bash
# Score fixes on the test pairs, after the main pipeline (needs work/saliency/*/orig.pkl).
# A: grayscale and matched brightness/contrast inputs (48,000 new Twitter-program calls).
# B: learned correction. The original setting is trained and evaluated on validation and test
#    (results/map_correction/). Four preset settings are trained, one is chosen on held-out
#    validation people by the largest hard-argmax gap and evaluated on test, and the original
#    setting is retrained with seeds 44 and 45 (seeds set the starting weights and the split of
#    validation people into fitting and tuning halves) and evaluated on test.
# C: upper bound from equalized group scores (uses labels at crop time, not a method), on the
#    main data and, if scripts/run_fairface.sh has run, on FairFace.
# Every interval uses the 2,000-draw people-level bootstrap.
# Runtime on a 4-core Intel Mac: about 1 hour, most of it Twitter's program in step A.
set -euo pipefail
cd "$(dirname "$0")/.."
# Main run's folders, whatever the calling shell has set
unset CROPBIAS_RUN_DIR CROPBIAS_IMAGES CROPBIAS_PAIRS_TEST CROPBIAS_PAIRS_VAL \
      CROPBIAS_PAIR_SEED_TEST CROPBIAS_PAIR_SEED_VAL CROPBIAS_SPLIT_SEED
PY=.venv/bin/python
MC="$PY -m scripts.map_correction_experiment"
B=2000

for v in gray matched_luminance; do
  $MC preprocess --split test --variant "$v" --workers 8 --bootstrap-iters $B
done
$MC train --epochs 12                     # the original objective and epoch choice
$MC eval --split val --bootstrap-iters $B
$MC eval --split test --bootstrap-iters $B
mkdir -p results/map_correction
cp work/map_correction/evaluation/test/summary.csv results/map_correction/summary_test.csv
cp work/map_correction/evaluation/val/summary.csv results/map_correction/summary_val.csv
cp work/map_correction/model/train_history.csv results/map_correction/train_history.csv
for obj in mean_square maximin; do
  for T in 0.2 0.05; do
    $MC train --epochs 12 --objective "$obj" --temperature "$T" --select hard \
      --output "work/map_correction/grid/${obj}_T${T}"
  done
done
$PY -m scripts.score_fixes select
chosen=$(cat work/map_correction/grid/chosen.txt)
$MC eval --split test --checkpoint "work/map_correction/grid/$chosen/best.pt" \
  --output work/map_correction/evaluation_selected/test --bootstrap-iters $B
for s in 44 45; do
  $MC train --epochs 12 --seed "$s" --output "work/map_correction/seeds/seed$s"
  $MC eval --split test --checkpoint "work/map_correction/seeds/seed$s/best.pt" \
    --output "work/map_correction/evaluation_seed$s/test" --bootstrap-iters $B
done
$PY -m scripts.score_fixes bound
$PY -m scripts.score_fixes summary

# The same bound on FairFace, if scripts/run_fairface.sh has run
if [ -f work/fairface/work/eval/val_desc.csv ]; then
  export CROPBIAS_RUN_DIR=work/fairface CROPBIAS_IMAGES=work/fairface/images
  export CROPBIAS_PAIRS_TEST=2000 CROPBIAS_PAIRS_VAL=500
  $PY -m scripts.score_fixes bound
  $PY -m scripts.score_fixes summary
  mkdir -p results/fairface/score_fixes
  cp work/fairface/results/score_fixes/*.csv results/fairface/score_fixes/
fi
