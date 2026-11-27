# Budgeted face-aware cropping: local experiment

This add-on is **independent of the old face-preserving evaluator**. It reads the
trusted WikiCeleb paired-composite layout and cached Twitter saliency maps. It
does not train a model or execute the Twitter x86_64 binary. It works on Apple
Silicon with Python 3.10–3.12 and does not require PyTorch or AWS.

## 1. Place files in your existing repository

Copy these two files from the add-on archive **into the repository root**:

- `scripts/budgeted_face_aware.py`
- `tests/test_budgeted_face_aware.py`

It requires `cropbias/face_preserving.py`, `cropbias/rules.py`, etc. from the
**latest** group repository. If `cropbias/face_preserving.py` is missing, update
the repository first rather than manually creating a fake replacement.

## 2. Restore your group's private data

The ZIP named `归档(2).zip` is flat: its `orig.pkl` is the **test** cache (24,000
pairs); its `orig.pkl 2` is the **validation** cache (6,000 pairs). Restore:

```text
data/dataset.csv
data/pairs.csv
work/composites/layout_val.csv
work/composites/layout_test.csv
work/saliency/val/orig.pkl     # comes from 'orig.pkl 2'
work/saliency/test/orig.pkl    # comes from 'orig.pkl'
```

**Never upload `data/`, `work/`, original images or pickle files to a public
GitHub repository. Only unpickle files from trusted group members.**

## 3. Install only lightweight libraries (Apple Silicon friendly)

From the repository root in VS Code Terminal:

```bash
python3 --version
python3 -m venv .venv-face
source .venv-face/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy>=1.24,<2' 'pandas>=2,<2.3' 'Pillow>=10.2,<11' 'tqdm>=4.60,<5' 'pytest>=7,<9'
```

Do **not** reinstall `requirements.txt` solely for this experiment: it contains
PyTorch and DeepGaze packages that are not needed to analyze saved maps.

## 4. Run these commands (from repo root)

```bash
# Check code and data paths with a tiny pilot (6 comparisons x 10 examples):
python -m pytest tests/test_budgeted_face_aware.py -q
python -m scripts.budgeted_face_aware --split val --limit-per-comp 10

# Full validation sweep, 6,000 examples, NO bootstrap or test-set tuning:
python -m scripts.budgeted_face_aware --split val --sweep

# Freeze the parameters; then ONE held-out test run, 24,000 examples:
python -m scripts.budgeted_face_aware --split test --delta 0.03 --fallback-budget 0.10 --min-face-px 32 --bootstrap-iters 2000
```

Outputs are aggregate CSVs under `results/budgeted_face_aware/`, with parameter
settings in their filenames, e.g.:

- `summary_val_sweep_px32.csv`
- `summary_test_d030_b100_px32.csv`
- `comparison_test_d030_b100_px32.csv`
- `bootstrap_test_d030_b100_px32.csv`

No pair-level identifiers are written by default. You can optionally use
`--save-per-pair` to write **private** outputs to `work/budgeted_face_aware/`.

## 5. What the methods actually do

The baseline is the original Twitter Argmax square crop. A **BOTH-square**
replacement is made only if the original crop is not BOTH and the best square
that keeps at least 90% of both faces' horizontal extents has Twitter saliency
mass >= `(1 - delta) * argmax_saliency_mass`. The best square is chosen by
existing `cropbias.face_preserving.choose`, which searches pixel-aligned square
windows. The four validation composites 1–2 px narrower than their height
cannot have a full-height square crop and keep the baseline.

The additional **fallback** fits the entire composite into a padded 256x256
preview. Only remaining not-BOTH images with both face widths projected to at
least `min-face-px` are eligible; at most `floor(fallback_budget * N)` of the
whole batch are selected, by descending minimum projected face width, with
pair-ID tie-break. No race or gender labels are accessed when choosing crops.

**Important**: exact full-image fallback *budgeting* is a batch-level offline
simulation. It is not yet a deployable per-image streaming rule. A deployment
study would calibrate a fixed geometry threshold on validation and apply that
threshold independently to each test image (budget then becomes an expected
rate, not an exact cap).

We report six signed pairwise exposure gaps, their largest absolute gap,
within-race gender gap (average of BF–BM and WF–WM), and gender-matched race
gap (average of WF–BF and WM–BM). The gender and race summaries are averages
of the signed pairwise gaps. A `BOTH` or `NEITHER` outcome contributes zero to
each corresponding pairwise gap. People-level CIs use the repository's
corrected two-way person bootstrap; the fallback decisions stay fixed during
bootstrap.

We also report the fraction displaying BOTH faces, projected **minimum** face
width in the 256-pixel preview, the fraction with both faces' projected widths
>= 32 px, and Twitter saliency retained **only on actual square crops**.
Full-image fallback is deliberately **excluded** from mean crop saliency:
calling full-image display 100% content-retention does **not** imply equivalent
readability/quality. Face width is only a geometric proxy, not a user study.

There is **no guarantee** that reducing a gender gap also reduces race gap, or
that this policy beats random switching at the same intervention rate. Compare
both metrics and existing random/sampling baselines before claiming a superior
mitigation method. The test data should not be reused to choose `delta` or the
fallback budget after inspecting outcomes.
