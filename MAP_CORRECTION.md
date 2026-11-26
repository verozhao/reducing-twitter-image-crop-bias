# Paired preprocessing and learned saliency correction

These are **new experiments** on the existing dataset. They do not change the earlier rule comparisons or their reported results. Twitter's inference model remains frozen; the training experiment fits a small residual network on top of its saved saliency grids. Evaluation still places a square crop at the highest point in the resulting map. A reduced demographic gap is an outcome to measure, not a guaranteed effect.

## Set up

From the repository root, create the environment as in `README.md`, download the Twitter program, and run the existing data pipeline. If you already have `data/dataset.csv`, `data/pairs.csv`, `work/images/`, `work/composites/{val,test}/`, `work/composites/layout_{val,test}.csv`, and `work/saliency/{val,test}/orig.pkl`, reuse them. The published zip deliberately excludes the original photos and private person/pair tables; therefore, training cannot start from the zip alone.

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
./scripts/get_twitter_model.sh
# Only if your prior local data/work folders are unavailable:
CROPBIAS_USER_AGENT='your-tool/0.1 (your contact)' CROPBIAS_DL_WORKERS=2 ./scripts/run_all.sh
```

`run_all.sh` also evaluates the existing decision rules. To prepare just this experiment with existing photos, use `python -m cropbias pairs`, `python -m cropbias composites`, `python -m cropbias saliency` as needed (see README for prior stages). Never rerun `pairs` if you need to retain earlier exact pair IDs and cached maps.

## A. Diagnose image preprocessing

```bash
.venv/bin/python -m scripts.map_correction_experiment preprocess --split val --variant gray
.venv/bin/python -m scripts.map_correction_experiment preprocess --split val --variant matched_luminance
```

- `gray`: convert *both entire source photos* to grayscale; this diagnoses sensitivity to color, not a proposed way to display people.
- `matched_luminance`: shift/scale the luminance of each entire source photo to the shared mean and standard deviation; RGB chroma is otherwise retained, with clipping. This changes skin appearance and may introduce artifacts. It is a **diagnostic** of lighting confounding, not a deployment recommendation or causal isolation of skin tone.
- Every transformed composite keeps the same dimensions, pair, and group-1 left/right assignment; each changed image is scored by Twitter's original program.
- The comparison is with the already stored original JPEG. Re-encoding a transformed image can itself change scores slightly; confirm large effects with paired visual inspection. Neither transformation is guaranteed to reduce bias.

Results: `work/map_correction/preprocess_eval/val/<variant>/summary.csv` and `per_pair.csv`. `summary.csv` reports six gaps, paired gap changes with two-way people-level bootstrap 95% intervals, crop outcome changes, no-face rate, and quality proxies. It uses the original model's saliency mass **for either chosen crop**; this proxy favors the baseline. When the earlier DeepGaze IIE density cache exists, its retained density is also reported, calculated on exactly the same subset.

Choose any preprocessing variant *using validation only*, then run the unchanged test people:

```bash
.venv/bin/python -m scripts.map_correction_experiment preprocess --split test --variant gray
.venv/bin/python -m scripts.map_correction_experiment preprocess --split test --variant matched_luminance
```

To check prerequisites and scoring on a short sample first: append `--limit 20` to a preprocessing command; output summary from so few pairs is only a functional check. Scoring all test pairs requires 24,000 new Twitter-program calls per variant (4,000 pairs x six comparisons) and may take hours.

## B. Train a corrected saliency map

```bash
.venv/bin/python -m scripts.map_correction_experiment train --epochs 12 --device cpu
.venv/bin/python -m scripts.map_correction_experiment eval --split val
.venv/bin/python -m scripts.map_correction_experiment eval --split test
```

`--device cuda` or `--device mps` can be used with a compatible PyTorch installation. The `train` command consumes **original** images and Twitter grids, independent of the preprocessing experiment. It splits existing validation people 50/50 within each of BF, BM, WF, WM into fitting and epoch-selection sets; only pairs with both people in the same half are used. Cross-half pairs are discarded. Test people are never used during training or epoch selection. Small `--limit` training pilots may fail because they cannot furnish all six comparison groups on both person-disjoint halves.

The correction network sees an RGB composite and Twitter's original saliency map at a fixed 32 x 128 analysis resolution and outputs a bounded spatial correction. Its final crop is still chosen by **hard argmax on the corrected map at Twitter's original grid resolution**, followed by the original square crop geometry. The network receives neither demographic labels nor face boxes at inference. At training time, the six pair types and face boxes define a soft proxy of the *exclusive-face* gap; the loss penalizes the mean signed gap within each type. Squared correction size and spatial roughness discourage destroying Twitter's original map. The hyperparameters are experimental, not tuned results.

The network is **not** the original Twitter network retrained end to end. With no human eye-fixation labels for the project composites, the objective cannot prove that the corrected map is a better prediction of human attention. The original-map fidelity loss and independent DeepGaze quality check are proxies only. If new eye-tracking labels become available, add a fixation prediction loss and report its held-out performance.

Outputs: `work/map_correction/model/best.pt` and `train_history.csv`; evaluation writes `work/map_correction/evaluation/{val,test}/summary.csv` and `per_pair.csv`. Model selection uses only the withheld validation people. The test CSV reports the same six comparisons on the **same original test pairs**, changes in outcomes, no-face crops, Twitter saliency retained, and DeepGaze IIE retained if its previous cache exists. The test comparison does not switch to the face-preserving fallback.

For an ablation, vary one training weight at a time and use distinct `--output` directories, e.g. `--fairness 0` for a zero-fairness control. Evaluate with `--checkpoint work/map_correction/model_0/best.pt`; do not select a favorite from repeated test evaluations. The zero-fairness, zero-initialized network with nonnegative fidelity/smoothness penalties should leave the original saliency map unchanged.

## Interpretation and caveats

- Report all six pair types. The current replication has a stronger and more consistent gender gap than its within-gender race comparisons; do not assert a White-favored race gap based on an insignificant or direction-changing estimate.
- Test samples reuse people across many pairs; bootstrap resamples people independently within each group and reweights each pair by the product of its two people's counts.
- This proof of concept uses Twitter's frozen saliency model plus a trained correction. A gain here would justify studying a trainable fixation model with a joint fixation and fairness objective; it would not demonstrate the causal source of Twitter's bias.
- DeepGaze IIE is an independent **predicted** gaze measure, not human fixation ground truth. For stronger claims, gather human preference/fixation data or use a genuinely independent fixation benchmark.
- The resized fixed-grid network sees variable-width composites warped to a common width. This makes training practical, but the interpolation could affect predictions; check hard-argmax test performance, not just the training loss.

## Relevant research

- Yee, Tantipongpipat, and Mishra (2021), [Image Cropping on Twitter](https://arxiv.org/abs/2105.08667): audit and caution about argmax bias; calls contrast a possible, unproven cause.
- Birhane, Prabhu, and Whaley (2022), [Auditing Saliency Cropping Algorithms](https://openaccess.thecvf.com/content/WACV2022/papers/Birhane_Auditing_Saliency_Cropping_Algorithms_WACV_2022_paper.pdf): demonstrates sensitivity to lighting/background and image construction, motivating paired preprocessing diagnostics.
- Huang et al. (2015), [SALICON: Reducing the Semantic Gap](https://openaccess.thecvf.com/content_iccv_2015/papers/Huang_SALICON_Reducing_the_ICCV_2015_paper.pdf): saliency networks trained against attention data; motivation for a fixation-quality objective, *not* evidence for the proposed fairness loss.
- Hanel et al. (2022), [Enhancing Fairness of Visual Attribute Predictors](https://openaccess.thecvf.com/content/ACCV2022/papers/Hanel_Enhancing_Fairness_of_Visual_Attribute_Predictors_ACCV_2022_paper.pdf): example of fairness regularization for image attribute prediction. The present loss is an adaptation to cropping and has not been validated by that paper.
