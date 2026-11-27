"""Square-stage control: is explicitly preserving BOTH faces better than moving
an identical set of crops to randomly sampled saliency-feasible square windows?

The image set, crop width, changed-pair mask, and relative saliency floor are
held fixed. No fallback or demographic labels are used in crop selection.

Two controls are evaluated:
  * uniform_feasible: uniform over all non-argmax, pixel-aligned square windows
    meeting the same 3% relative saliency-retention floor.
  * uniform_quality_matched: uniform over the subset additionally within +/-
    0.5 percentage points of the face-aware selected window's saliency mass.

Their *expected* exposure is calculated analytically from the candidates.
Monte Carlo draws quantify decision randomness conditional on the test images.
A paired two-way people bootstrap compares expected group gaps, conditional
on the already chosen intervention mask and known faces.

Run from repository root:
  python -m scripts.square_stage_random_control --split val --draws 200
  python -m scripts.square_stage_random_control --split test --draws 500 --bootstrap-iters 1000

Results are aggregate-only CSVs under results/square_stage_random_control/.
The private data, per-image choices, and source photo identifiers are not saved.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cropbias import config as C
from cropbias.geometry import BOTH, inside_fraction
from cropbias.stats import composite_weights, person_weights
from cropbias.twitter_saliency import load_store
from scripts.budgeted_face_aware import COMPS, apply_policy, load_records

CONTROL_NAMES = ("uniform_feasible", "uniform_quality_matched")


def _gap_values(contrib: np.ndarray, comp_idx: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Identical six-comparison signed parity convention to budgeted_face_aware."""
    means = np.array([np.mean(contrib[comp_idx == j]) for j in range(len(COMPS))])
    gender = 50.0 * (means[COMPS.index("BF-BM")] + means[COMPS.index("WF-WM")])
    race = 50.0 * (means[COMPS.index("WF-BF")] + means[COMPS.index("WM-BM")])
    return float(gender), float(race), float(100 * np.max(np.abs(means))), means


def _random_candidates(W: int, H: int, face1: tuple[float, float],
                       face2: tuple[float, float], xs: np.ndarray,
                       saliency: np.ndarray, original_x0: int,
                       original_quality: float, faceaware_quality: float,
                       delta: float, tolerance: float) -> dict:
    """Prepare all *non-original* quality-feasible pixel-aligned square windows.

    Returns two candidate pools (and their expected metrics). Rejecting the
    original window makes the *number of actual crop edits* identical to the
    deterministic face-aware policy, not merely the number of eligible images.
    """
    if W < H or H <= 0:
        raise ValueError("Intervened image must admit a square crop")
    if xs.ndim != 1 or saliency.ndim != 2 or saliency.shape[1] != len(xs):
        raise ValueError("Malformed saliency grid")
    if not np.all(np.diff(xs) >= 0) or not np.isfinite(saliency).all():
        raise ValueError("Unsorted/nonfinite saliency grid")

    # Exactly the same normalized saliency mass as View and choose().
    values = saliency.astype(np.float64)
    z = np.exp(values - np.max(values))
    masses = (z / z.sum()).sum(axis=0)
    starts = np.arange(W - H + 1, dtype=np.int32)
    prefix = np.r_[0.0, np.cumsum(masses)]
    left = np.searchsorted(xs, starts, side="left")
    right = np.searchsorted(xs, starts + H, side="left")
    quality = prefix[right] - prefix[left]
    if abs(quality[original_x0] - original_quality) > 2e-7:
        raise ValueError("Stored original quality does not match saliency cache")

    f1 = inside_fraction(face1, starts, starts + H) >= C.KEEP_FRACTION
    f2 = inside_fraction(face2, starts, starts + H) >= C.KEEP_FRACTION
    outcomes = (f1.astype(np.int8) + 2 * f2.astype(np.int8))
    min_width = np.minimum(
        np.maximum(0.0, np.minimum(face1[1], starts + H) - np.maximum(face1[0], starts)),
        np.maximum(0.0, np.minimum(face2[1], starts + H) - np.maximum(face2[0], starts)),
    )
    if not np.any((outcomes == BOTH) & (np.abs(quality - faceaware_quality) <= 2e-7)):
        raise ValueError("Face-aware selected quality has no BOTH window; check data alignment")
    allowed = ((quality >= (1.0 - delta) * original_quality - 1e-12)
               & (starts != original_x0))
    matched = allowed & (np.abs(quality - faceaware_quality) <= tolerance + 1e-12)
    if not allowed.any() or not matched.any():
        raise ValueError("Missing non-original quality-feasible/matched alternative")

    def pool(mask: np.ndarray) -> dict:
        o = outcomes[mask]
        q = quality[mask]
        w = min_width[mask]
        counts = np.bincount(o, minlength=4)
        return {
            "outcomes": o,
            "quality": q,
            "width": w,
            "expected_gap": float(0.5 * (counts[1] - counts[2]) / len(o)),
            "prob_both": float(counts[BOTH] / len(o)),
            "prob_width32": float(np.mean(w >= 32.0)),
            "expected_width": float(w.mean()),
            "expected_quality": float(q.mean()),
            "n_candidates": len(o),
        }

    return {"uniform_feasible": pool(allowed),
            "uniform_quality_matched": pool(matched)}


def build_control_records(pairs: pd.DataFrame, rec: pd.DataFrame,
                          saliency: dict, layout: pd.DataFrame,
                          delta: float, tolerance: float,
                          draws: int, seed: int) -> tuple[dict, dict]:
    """Enumerate alternatives only for pairs Face-aware actually changes."""
    sq = apply_policy(rec, delta, 0.0, min_face_px=32.0)
    if np.any(sq["mode"].to_numpy() == 2):
        raise AssertionError("Square-stage experiment may not use fallback")
    selected = np.flatnonzero(sq["mode"].to_numpy() == 1)
    n = len(rec)
    methods = {}
    original_y = sq["arg_gap_contribution"].to_numpy(dtype=float)
    square_y = sq["gap_contribution"].to_numpy(dtype=float)
    original_q = rec["arg_quality"].to_numpy(dtype=float)
    square_q = sq["square_quality"].to_numpy(dtype=float)
    square_width = sq["min_displayed_face_px"].to_numpy(dtype=float)
    square_both = (sq["outcome"].to_numpy() == BOTH).astype(float)
    for name in CONTROL_NAMES:
        methods[name] = {
            "gap": original_y.copy(),
            "quality": original_q.copy(),
            "width": rec["arg_min_face_px"].to_numpy(dtype=float).copy(),
            "both": (rec["arg_outcome"].to_numpy() == BOTH).astype(float).copy(),
            "p_width32": (rec["arg_min_face_px"].to_numpy(dtype=float) >= 32).astype(float).copy(),
            "n_candidates": [],
            "draw_change": np.zeros((draws, n), dtype=np.float32),
        }
    rng = np.random.default_rng(seed)
    for count, ix in enumerate(selected):
        if count and count % 2500 == 0:
            print(f"Enumerated {count}/{len(selected)} face-aware interventions", flush=True)
        pid = str(rec.at[ix, "pair_id"])
        row = layout.loc[pid]
        W, H = int(row.W), int(row.H)
        pools = _random_candidates(W, H,
              (float(row.f1x0), float(row.f1x1)),
              (float(row.f2x0), float(row.f2x1)),
              np.asarray(saliency[pid].xs), np.asarray(saliency[pid].S),
              int(rec.at[ix, "arg_x0"]), float(rec.at[ix, "arg_quality"]),
              float(rec.at[ix, "best_both_quality"]), delta, tolerance)
        for name, pool in pools.items():
            state = methods[name]
            state["gap"][ix] = pool["expected_gap"]
            state["quality"][ix] = pool["expected_quality"]
            state["width"][ix] = pool["expected_width"]
            state["both"][ix] = pool["prob_both"]
            state["p_width32"][ix] = pool["prob_width32"]
            state["n_candidates"].append(pool["n_candidates"])
            idx = rng.integers(0, pool["n_candidates"], size=draws)
            o = pool["outcomes"][idx]
            state["draw_change"][:, ix] = 0.5 * ((o == 1).astype(float) - (o == 2).astype(float))

    methods["argmax"] = {
        "gap": original_y, "quality": original_q,
        "width": rec["arg_min_face_px"].to_numpy(dtype=float),
        "both": (rec["arg_outcome"].to_numpy() == BOTH).astype(float),
        "p_width32": (rec["arg_min_face_px"].to_numpy(dtype=float) >= 32).astype(float),
    }
    methods["face_aware_square"] = {
        "gap": square_y, "quality": square_q, "width": square_width,
        "both": square_both, "p_width32": (square_width >= 32).astype(float),
    }
    return methods, {"changed_indices": selected, "square": sq}


def method_summary(methods: dict, selected: np.ndarray, pairs: pd.DataFrame,
                   rec: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    comp_idx = pd.Categorical(pairs.comparison, categories=COMPS).codes
    if (comp_idx < 0).any():
        raise ValueError("Unexpected comparison name")
    rows = []
    qf = methods["face_aware_square"]["quality"]
    for name, state in methods.items():
        gender, race, maximum, _ = _gap_values(state["gap"], comp_idx)
        row = {"method": name, "pairs": len(pairs), "gender_gap_pp": gender,
               "race_matched_gap_pp": race, "max_six_abs_gap_pp": maximum,
               "actual_square_edit_pct": 0.0 if name == "argmax" else 100 * len(selected) / len(pairs),
               "fallback_pct": 0.0,
               "both_face_pct": 100.0 * float(np.mean(state["both"])),
               "two_faces_at_least_32px_pct": 100.0 * float(np.mean(state["p_width32"])),
               "mean_min_displayed_face_px": float(np.mean(state["width"])),
               "mean_twitter_saliency_retained_pct": 100 * float(np.mean(state["quality"])),
               "mean_quality_on_changed_pct": 100 * float(np.mean(state["quality"][selected]))
                    if len(selected) else np.nan,
               "mean_abs_quality_mismatch_to_faceaware_on_changed_pp":
                    100 * float(np.mean(np.abs(state["quality"][selected] - qf[selected])))
                    if len(selected) else np.nan,
               "quality_tolerance_pp": 100 * tolerance,
               "mean_candidate_pool_size": float(np.mean(state.get("n_candidates", [])))
                    if "n_candidates" in state else np.nan,
               "pct_changed_pairs_with_single_candidate":
                    100 * float(np.mean(np.asarray(state.get("n_candidates", [])) == 1))
                    if "n_candidates" in state and len(selected) else np.nan,
               }
        rows.append(row)
    return pd.DataFrame(rows)


def per_comparison_summary(methods: dict, pairs: pd.DataFrame) -> pd.DataFrame:
    """Expose subgroup-specific regressions hidden by single aggregate metrics."""
    groups = pairs.comparison.to_numpy()
    rows = []
    for method, state in methods.items():
        for c in COMPS:
            m = groups == c
            if not np.any(m):
                raise ValueError(f"Missing comparison {c}")
            rows.append({"method": method, "comparison": c, "pairs": int(m.sum()),
                         "signed_gap_pp": 100 * float(np.mean(state["gap"][m])),
                         "both_face_pct": 100 * float(np.mean(state["both"][m])),
                         "mean_twitter_saliency_retained_pct":
                             100 * float(np.mean(state["quality"][m]))})
    return pd.DataFrame(rows)


def random_draw_summary(methods: dict, selected: np.ndarray,
                        pairs: pd.DataFrame) -> pd.DataFrame:
    comp_idx = pd.Categorical(pairs.comparison, categories=COMPS).codes
    result = []
    # Accumulate per-comparison contributions over only randomly changed pairs.
    for name in CONTROL_NAMES:
        state = methods[name]
        baseline = methods["argmax"]["gap"]
        base_by_group = np.array([baseline[comp_idx == j].sum() for j in range(6)])
        den = np.bincount(comp_idx, minlength=6)
        totals = np.tile(base_by_group[None, :], (state["draw_change"].shape[0], 1))
        # Grouped sums for changed pairs only; each draw has exactly the same changed set.
        for j in range(6):
            ix = selected[comp_idx[selected] == j]
            if len(ix):
                totals[:, j] += (state["draw_change"][:, ix].sum(axis=1)
                                 - baseline[ix].sum())
        means = totals / den[None, :]
        gender = 50 * (means[:, COMPS.index("BF-BM")] + means[:, COMPS.index("WF-WM")])
        race = 50 * (means[:, COMPS.index("WF-BF")] + means[:, COMPS.index("WM-BM")])
        maxgap = 100 * np.max(np.abs(means), axis=1)
        for i in range(len(gender)):
            result.append({"method": name, "draw": i,
                           "gender_gap_pp": float(gender[i]),
                           "race_matched_gap_pp": float(race[i]),
                           "max_six_abs_gap_pp": float(maxgap[i])})
    return pd.DataFrame(result)


def paired_bootstrap(pairs: pd.DataFrame, methods: dict,
                     iters: int) -> pd.DataFrame:
    """People-clustered CI for Face-aware minus *expected* random policy.

    This holds intervention eligibility and policy decisions fixed, matching
    the existing project bootstrap convention. Random-decision variability is
    separately quantified by random_draw_summary().
    """
    split = str(pairs.split.iloc[0])
    dataset = pd.read_csv(C.DATA / "dataset.csv")
    people = {g: np.array(sorted(dataset.loc[
        (dataset.group == g) & (dataset.split == split), "qid"])) for g in C.GROUPS}
    weights = composite_weights(pairs, people, person_weights(people, iters, C.BOOTSTRAP_SEED))
    comp = pairs.comparison.to_numpy()
    den = []
    for c in COMPS:
        den.append(weights[:, comp == c].sum(axis=1, dtype=np.float64))

    def gap_draw(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = []
        for j, c in enumerate(COMPS):
            mask = comp == c
            num = weights[:, mask].astype(np.float64) @ y[mask]
            means.append(np.divide(num, den[j], out=np.full(iters, np.nan), where=den[j] > 0))
        v = np.stack(means)
        return (50 * (v[COMPS.index("BF-BM")] + v[COMPS.index("WF-WM")]),
                50 * (v[COMPS.index("WF-BF")] + v[COMPS.index("WM-BM")]),
                100 * np.max(np.abs(v), axis=0))

    reference = gap_draw(methods["face_aware_square"]["gap"])
    names = ("gender_gap_pp", "race_matched_gap_pp", "max_six_abs_gap_pp")
    rows = []
    for m in CONTROL_NAMES:
        alt = gap_draw(methods[m]["gap"])
        for j, label in enumerate(names):
            diff = reference[j] - alt[j]
            diff = diff[np.isfinite(diff)]
            rows.append({"comparison": "face_aware_minus_" + m,
                         "metric": label,
                         "estimate_kind": "CI for difference in population-level expected gap, fixed decisions",
                         "ci_low_95": np.percentile(diff, 2.5) if len(diff) else np.nan,
                         "ci_high_95": np.percentile(diff, 97.5) if len(diff) else np.nan,
                         "valid_bootstrap_draws": len(diff)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--delta", type=float, default=.03)
    ap.add_argument("--quality-tolerance", type=float, default=.005,
                    help="abs saliency-mass tolerance; .005 is 0.5 percentage points")
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--seed", type=int, default=271828)
    ap.add_argument("--bootstrap-iters", type=int, default=0)
    ap.add_argument("--limit-per-comp", type=int, default=None)
    args = ap.parse_args()
    if not (0 <= args.delta < 1 and args.quality_tolerance >= 0
            and args.draws >= 2 and (args.bootstrap_iters == 0 or args.bootstrap_iters >= 50)):
        ap.error("invalid arguments")
    pairs, rec = load_records(args.split, args.limit_per_comp)
    saliency = load_store(args.split, "orig")
    layout = pd.read_csv(C.WORK / "composites" / f"layout_{args.split}.csv").set_index("pair_id")
    print("Enumerating quality-feasible alternatives", flush=True)
    methods, extra = build_control_records(pairs, rec, saliency, layout,
                                            args.delta, args.quality_tolerance,
                                            args.draws, args.seed)
    summary = method_summary(methods, extra["changed_indices"], pairs, rec, args.quality_tolerance)
    draws = random_draw_summary(methods, extra["changed_indices"], pairs)
    outdir = C.RESULTS / "square_stage_random_control"
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "_pilot" if args.limit_per_comp else ""
    prefix = f"{args.split}_d{round(args.delta*1000):03d}_tol{round(args.quality_tolerance*1000):03d}{suffix}"
    summary.to_csv(outdir / f"summary_{prefix}.csv", index=False)
    per_comparison_summary(methods, pairs).to_csv(outdir / f"comparisons_{prefix}.csv", index=False)
    draws.to_csv(outdir / f"random_draws_{prefix}.csv", index=False)
    quantiles = draws.groupby("method").agg(
        gender_mean=("gender_gap_pp", "mean"), gender_q025=("gender_gap_pp", lambda x: x.quantile(.025)),
        gender_q975=("gender_gap_pp", lambda x: x.quantile(.975)),
        maxgap_mean=("max_six_abs_gap_pp", "mean"), maxgap_q025=("max_six_abs_gap_pp", lambda x: x.quantile(.025)),
        maxgap_q975=("max_six_abs_gap_pp", lambda x: x.quantile(.975)),
    ).reset_index()
    quantiles.to_csv(outdir / f"random_distribution_{prefix}.csv", index=False)
    if args.bootstrap_iters:
        if args.limit_per_comp:
            print("Skipping research bootstrap for smoke test")
        else:
            ci = paired_bootstrap(pairs, methods, args.bootstrap_iters)
            ci.to_csv(outdir / f"paired_bootstrap_{prefix}.csv", index=False)
            print("Paired bootstrap CIs for face-aware minus random expected gaps:")
            print(ci.to_string(index=False))
    print("\nExpected outcomes (random policies are averaged analytically):")
    print(summary[["method", "gender_gap_pp", "race_matched_gap_pp",
        "max_six_abs_gap_pp", "both_face_pct", "actual_square_edit_pct",
        "mean_twitter_saliency_retained_pct", "mean_abs_quality_mismatch_to_faceaware_on_changed_pp"
    ]].to_string(index=False))
    print("\nConditional random-draw distribution:")
    print(quantiles.to_string(index=False))
    print("Random policies intervene on precisely the SAME pairs as face-aware. No fallback is used.")
    if args.split == "test":
        print("This is a secondary, already-inspected test-set analysis, not fresh holdout confirmation.")
    else:
        print("Validation-only result; control protocol should be frozen before interpreting test results.")
    print(f"Saved {outdir}")


if __name__ == "__main__":
    main()
