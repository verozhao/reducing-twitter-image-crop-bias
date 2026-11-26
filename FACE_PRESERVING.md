# Face-preserving square preview

This add-on uses the existing Twitter saliency model and the saved composite face
boxes. It does **not** train a new model, change any of the existing crop rules,
or overwrite the existing `results/square/` tables.

For every wide, height-256 composite it searches all integer square positions.
If a square keeps at least 90% of **each** detected face's horizontal box width,
it takes the square retaining the most Twitter saliency mass. If no such square
exists, the preview fits the entire composite into a 256-by-256 square and fills
the unused space with neutral padding. Candidate squares tied on saliency are
ordered by proximity to the original argmax crop. Neither race nor gender is an
input to the rule.

## Dependencies and data

Run commands from the extracted repository root. This archive intentionally
does not contain the private `data/` and `work/` directories, the Commons images,
or Twitter's model binary. With a prepared copy of this project, restore the
corresponding `data/` and `work/` folders; make sure `data/dataset.csv`,
`data/pairs.csv`, `work/composites/layout_test.csv`, and
`work/saliency/test/orig.pkl` exist. The new evaluation reuses these files;
DeepGaze scores in `work/deepgaze/test.pkl` are optional.

For a fresh run, the repository's original setup instructions apply:

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
./scripts/get_twitter_model.sh
export CROPBIAS_USER_AGENT='your-tool/0.1 (your contact URL or email)'
export CROPBIAS_DL_WORKERS=2
.venv/bin/python -m cropbias people
.venv/bin/python -m cropbias download --small-originals
.venv/bin/python -m cropbias faces
.venv/bin/python -m cropbias dataset
.venv/bin/python -m cropbias pairs
.venv/bin/python -m cropbias composites
.venv/bin/python -m cropbias saliency --workers 8
```

The existing pipeline splits people into validation/test and balances positions
when it makes the pairs. Twitter's released program supplies the unchanged
saliency scores. No annotated training data, checkpoint, optimizer, or fitting
step is needed for the new rule.

## Evaluate and view previews

```bash
.venv/bin/python -m scripts.face_preserving_eval --split test --render 24
```

Results:

- `results/face_preserving/summary_test.csv`: group gaps, 95% people-level
  bootstrap intervals for the argmax gap and fallback rate, square feasibility,
  and crop/preview quality measurements.
- `work/eval_face_preserving/test/per_pair.csv`: private pair-level choices and
  measurements. Do not publish this without checking person identifiers.
- `work/eval_face_preserving/test/previews/`: the first 24 actual square display
  images, including the padded full-image fallbacks.

Run `--render 0` to skip preview images, or `--render -1` to render every pair.
To check the validation split, use `--split val`. For a quick pilot after the
pair tables exist, run the existing `composites --limit 20` and
`saliency --limit 20` stages, then this new runner with `--limit 20`.
Keep pilot results separate from the full test results (the filename includes
`pilot20`). Existing saved evaluation results are not required.

Verify the rule's geometry with:

```bash
.venv/bin/python -m pytest tests/test_face_preserving.py
```

## Interpreting the result

The new exclusive-face gap is **zero by construction** whenever both detected
face boxes are valid, because every preview keeps both faces. This is a display
policy result, not evidence that saliency scores became unbiased. Report the
fallback rate and image scale alongside the gap: when faces cannot fit in one
square, the full image is smaller within the square display area. The 100%
saliency-retained score assigned to fallback means that no original image
content was cut away. It does not measure readability of a reduced image.
DeepGaze scores are calculated only for the saved DeepGaze subset.

Face-detection misses and erroneous boxes are the main limitation. The existing
dataset excludes images with no clear detected face, so evaluation on these
pairs alone does not measure how the policy handles missed faces in the wild.
