#!/usr/bin/env bash
# Score fixes on the test pairs, after the main pipeline (needs work/saliency/*/orig.pkl).
# A: grayscale and matched brightness/contrast inputs (48,000 new Twitter-program calls).
# B: learned correction with its original settings, then four preset settings, the setting and epoch chosen on held-out
#    validation people by the largest hard-argmax gap, then one test evaluation.
# C: upper bound from equalized group scores (uses labels at crop time, not a method).
# Runtime on a 4-core Intel Mac: about 1 hour, most of it Twitter's program in step A.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MC="$PY -m scripts.map_correction_experiment"

for v in gray matched_luminance; do
  $MC preprocess --split test --variant "$v" --workers 8 --bootstrap-iters 2000
done
$MC train --epochs 12                     # the original objective and epoch choice
$MC eval --split test
for obj in mean_square maximin; do
  for T in 0.2 0.05; do
    $MC train --epochs 12 --objective "$obj" --temperature "$T" --select hard \
      --output "work/map_correction/grid/${obj}_T${T}"
  done
done
$PY -m scripts.score_fixes select
chosen=$(cat work/map_correction/grid/chosen.txt)
$MC eval --split test --checkpoint "work/map_correction/grid/$chosen/best.pt" \
  --output work/map_correction/evaluation_selected/test --bootstrap-iters 2000
$PY -m scripts.score_fixes bound
$PY -m scripts.score_fixes summary
