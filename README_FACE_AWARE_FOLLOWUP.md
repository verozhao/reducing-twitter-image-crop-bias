# Face-aware cropping — controlled follow-up (secondary analysis)

## Research question and scope

The existing budgeted face-aware policy first replaces Argmax with a high-saliency square crop retaining BOTH faces, then assigns up to 10% of the entire batch to full-image previews by ranking minimum face width. This is **an offline batch allocation**, not a per-image deployment rule.

This add-on evaluates two focused questions:

1. **Deployment robustness:** Can a fallback width threshold calibrated on validation be applied independently to each held-out test image? Report the actual test fallback rate instead of forcing it to 10%.
2. **Allocation ablation:** Holding the face-aware square stage, eligible set, and test fallback count fixed, does ranking by projected face width differ from uniform random allocation? This tests the **fallback assignment only**, NOT the effectiveness of the whole face-aware algorithm against generic random crops.

The main test set has already been inspected. These analyses are exploratory secondary robustness tests and should not be described as a fresh blind confirmation.

## Installation in your local Mac repository

Download `face_aware_followup_addon.zip` to `~/Downloads/`, open the existing project in VS Code, and activate its existing `.venv`:

```bash
unzip -o ~/Downloads/face_aware_followup_addon.zip -d .
source .venv/bin/activate
python -m pytest tests/test_face_aware_followup.py tests/test_budgeted_face_aware.py -q
```

The add-on creates these **new** files; it does not replace the earlier experiment:

```text
scripts/face_aware_followup.py
tests/test_face_aware_followup.py
README_FACE_AWARE_FOLLOWUP.md
```

It depends on `scripts/budgeted_face_aware.py` and the six private files already installed under `data/` and `work/`.

## Small smoke test

```bash
python -m scripts.face_aware_followup --limit-per-comp 10 --draws 10
```

The resulting pilot is only for debugging.

## Full secondary analysis

```bash
python -m scripts.face_aware_followup --delta 0.03 --fallback-budget 0.10 --min-face-px 32 --draws 200 --bootstrap-iters 1000
```

The script first loads validation (6000 image pairs) to calibrate a fixed width threshold, then test (24000 pairs) with that threshold frozen. It draws 200 fixed-seed random allocations from the **same eligible pool** with **exactly the same number of fallbacks as the online policy on test**. The square-crop decisions are unchanged in every draw.

Outputs in `results/face_aware_followup/`:

- `calibration.csv`: validation threshold, actual test fallback fraction, and number of eligible test images.
- `methods_test.csv`: Argmax, face-aware square-only, batch-budgeted fallback, validation-calibrated online, and average random eligible fallback.
- `random_fallback_distribution.csv`: randomization distribution (mean, SD, 2.5th and 97.5th percentiles), **not a people-level CI**.
- `random_fallback_draws.csv`: 200 per-draw aggregate results; no per-person IDs.
- `comparison_online_test.csv`: six pairwise online exposure gaps.
- `bootstrap_online_test.csv`: paired people-level bootstrap percentile CIs for the online policy (fixed threshold and decisions within resamples).

## Important methodological limits

- The online threshold is calibrated using validation only, but the experiment was motivated after inspecting the test results; it must be reported as exploratory robustness work. Full out-of-sample confirmation requires another untouched dataset/draw.
- The offline batch policy achieves exactly 10% fallback by ranking the whole test set; the online threshold may exceed 10% when the distribution shifts. If a **hard per-batch cap** is required, the online rule is not an exact replacement.
- The random comparison matches the square-stage actions, fallback **count**, and minimum readability eligibility. It does **not** match the distribution of actual projected face sizes. Differences in readability must therefore be reported; randomization is diagnostic, not an apples-to-apples quality-matched superiority test.
- The measured 256px face width is only a geometry proxy; it does not establish actual human readability. 3:4 portrait previews and independent quality or human perception remain untested here.
- Group labels are never used by either crop-selection policy, but they are used for fairness evaluation.
- `results/` files contain aggregate statistics only. Private person IDs, `data/`, `work/` and pickle caches must never be committed.

## Reference run on supplied original-seed data

On the supplied original-seed WikiCeleb test set, the validation threshold was approximately **48.445 px**; the online fallback rate was **11.696%**. The online policy had gender gap **5.700 pp** and max-six absolute gap **6.650 pp**. Across 200 random fallback allocations, the corresponding means were **5.737 pp** and **6.684 pp**. Differences are small and do not demonstrate a major fairness advantage of width ranking.
