# Square-stage constrained random controls

## Research question

Does *prioritizing a square crop that preserves both detected faces*, rather
than merely changing crop positions, improve demographic exposure fairness
when intervention count and Twitter saliency retention are comparable?

This is **not** a newly trained model. It reuses the original Twitter saliency
maps and the existing `budgeted_face_aware` decision rule. It does **not**
require the Twitter executable, an AWS instance, or a GPU.

## Design

We hold fixed: the 24,000 original-seed test composites; square crop geometry;
project-standard >=90% face-width inclusion criterion; a maximum 3% *relative*
loss in Twitter saliency retention from Argmax; the exact **same changed-pair
mask** as Face-aware square; no full-image fallback; and no demographic
attributes at selection time. Each alternative **must be a different
pixel-aligned square window** from the original Argmax crop.

- **Argmax**: original Twitter crop.
- **Face-aware square**: for each image with a quality-eligible square that
  retains *both faces*, choose the highest-Twitter-saliency such square.
- **Uniform feasible**: on *exactly those same changed images*, uniformly sample
  from all non-original squares satisfying the same relative saliency floor.
- **Uniform quality matched (primary control)**: uniformly sample instead from
  non-original squares satisfying the floor **and** having saliency retention
  within **0.005 absolute probability mass = 0.5 percentage points** of the
  Face-aware chosen square. The Face-aware square is permitted in this pool,
  which makes this a conservative control, not a forced non-BOTH comparison.

All candidate windows are enumerated at one-pixel horizontal intervals, using
the same normalized Twitter saliency mass as the repository `View` class. For
random policies, *expected per-image contributions* are evaluated exactly by
averaging all permitted positions; 500 seeded random full-dataset draws
separately quantify random-decision variation. The people-level two-way paired
bootstrap (1,000 draws) compares Face-aware to **expected random** exposure,
conditional on the selected pair set and decisions. These bootstrap intervals
**do not include new model retraining, dataset rebuilding, or repeated random
position draws**. The original test data has already been inspected, so this
is an **exploratory secondary analysis**, not fresh holdout confirmation.

### Important interpretation limits

1. The intervention mask is itself determined by the Face-aware feasibility
   rule. This experiment isolates the *crop-position choice conditional on
   those selected images*, **not** the benefit of independently choosing the
   set of images to intervene on. It does not cover all possible random-crop
   policies.
2. Preserving BOTH is explicitly the design objective, so higher
   both-face retention is partly definitional; the value of the comparison is
   to quantify the *resulting demographic gaps under quality-matched changes*.
3. Twitter saliency retention is a proxy for crop content coverage and not an
   independent human visual-quality judgment. Face boundaries depend on the
   upstream face detector, and group labels come from the study dataset.
4. Neither the aggregate expected gaps nor the paired bootstrap constitute
   independent external replication. There is no FairFace experiment here.
5. The `max_six_abs_gap_pp` of a randomized policy is computed from its
   **expected per-image parity contribution**. The draw-distribution CSV
   reports the nonlinear maximum gap for each realized random dataset draw;
   the two numbers need not be exactly identical.
6. No full-image fallback is used in this experiment. It should not be
   directly equated with the earlier online 10%-fallback policy.

## Local setup

From a recent repository branch containing `scripts/budgeted_face_aware.py`:

```
source .venv/bin/activate
python -m pytest tests/test_square_stage_random_control.py -q
```

The same **matched seed** must be used for all private inputs:

```
data/dataset.csv
data/pairs.csv
work/composites/layout_val.csv
work/composites/layout_test.csv
work/saliency/val/orig.pkl
work/saliency/test/orig.pkl
```

Then run:

```
python -m scripts.square_stage_random_control --split val --draws 300
python -m scripts.square_stage_random_control --split test --draws 500 --bootstrap-iters 1000
```

The default random seed is 271828; change with `--seed`. The experiment
parameters `--delta 0.03 --quality-tolerance 0.005` were fixed across splits.
Do **not** tune them after inspecting the test results.

Expected output under `results/square_stage_random_control/`:

- `summary_val_d030_tol005.csv` and `summary_test_d030_tol005.csv`
- `random_distribution_val_d030_tol005.csv` and
  `random_distribution_test_d030_tol005.csv`
- `comparisons_val_d030_tol005.csv` and `comparisons_test_d030_tol005.csv`
- `paired_bootstrap_test_d030_tol005.csv`
- `random_draws_*.csv` (repeated aggregate draws; optional to commit)

These CSVs contain only aggregate metrics and random-draw summaries, no
individual person records or private image/pair identifiers.

## Observed results (original seed)

| Test method | Gender gap (pp) | Max six absolute gap (pp) | Both faces (%) | Overall saliency retention (%) |
|---|---:|---:|---:|---:|
| Twitter Argmax | 10.619 | 12.038 | 1.363 | 73.549 |
| Uniform feasible | 9.032 | 10.688 | 11.642 | 76.632 |
| Uniform quality matched | 8.220 | 9.895 | 25.132 | 78.865 |
| Face-aware square | **7.112** | **8.150** | **40.092** | **78.874** |

All three interventions modify the exact same **9,295 / 24,000 = 38.729%**
original crops, with no full-image fallback. Among changed images, the
quality-matched control differs from Face-aware by **0.0437 percentage points
in average absolute retained saliency**; mean candidate pool size 14.70,
with 1.85% of changed cases having only one valid candidate.

Paired people-level 95% CI for **Face-aware minus uniform quality matched**:

- Gender gap: **[-1.726, -0.431] pp**.
- Maximum six absolute gap: **[-2.792, -0.416] pp**.
- Gender-matched race gap: **[-0.124, +1.255] pp**, includes zero.

Interpretation: On these data, explicit two-face preference shows improved
exposure parity over the quality-matched window-position control, under equal
changed-image budget and extremely close saliency retention. Evidence about
race-matched gap differences is inconclusive. These results do not establish
real-world visual quality or generalization to a new dataset.

## Repository hygiene

Commit **only** the new script, unit tests, this document, and selected
aggregate CSVs. Do not commit `data/`, `work/`, photos, faces or per-image
records. Inspect `git diff --cached --name-only` before pushing.
