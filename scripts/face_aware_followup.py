"""Scientific follow-up for budgeted face-aware cropping.

1. Calibrate a *fixed per-image fallback face-width threshold* on validation.
2. Apply it without batch quota to the test set.
3. Compare to randomized fallback allocations from the exact same eligible pool
   with the same NUMBER of fallback images; square-crop stage stays unchanged.

The randomization control isolates fallback *allocation/ranking*, not the causal
value of the whole face-aware method. The test set has already been inspected;
this is a secondary robustness analysis, not a new preregistered holdout test.

Run (in project root):
    python -m scripts.face_aware_followup --draws 200 --bootstrap-iters 1000
No model inference; does not use demographic labels for cropping decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import math

import numpy as np
import pandas as pd

from cropbias import config as C
from cropbias.geometry import BOTH
from scripts.budgeted_face_aware import (
    apply_policy, bootstrap, load_records, per_comparison, summarize,
)


def fallback_eligible(rec: pd.DataFrame, square_only: pd.DataFrame,
                      min_face_px: float) -> np.ndarray:
    """Same candidate pool as budgeted_face_aware.apply_policy()."""
    return ((square_only["mode"].to_numpy() == 0)
            & (rec["arg_outcome"].to_numpy() != BOTH)
            & (rec["full_min_face_px"].to_numpy(dtype=float) >= min_face_px))


def calibrate_threshold(rec_val: pd.DataFrame, delta: float,
                        budget: float, min_face_px: float) -> tuple[float, dict]:
    """Fit a single face-width threshold on validation ONLY.

    When the cutoff is tied, use >= and disclose the resulting val rate. For
    the supplied original-seed data the boundary is strictly separated.
    """
    if not (0 <= budget <= 1):
        raise ValueError("budget must be between 0 and 1")
    sq = apply_policy(rec_val, delta, 0.0, min_face_px)
    mask = fallback_eligible(rec_val, sq, min_face_px)
    widths = rec_val.loc[mask, "full_min_face_px"].to_numpy(dtype=float)
    k = min(int(math.floor(budget * len(rec_val) + 1e-10)), len(widths))
    if k == 0:
        threshold = math.inf
    elif k == len(widths):
        threshold = float(np.nextafter(widths.min(), -math.inf))
    else:
        descending = np.sort(widths)[::-1]
        hi = descending[k - 1]
        lo = descending[k]
        threshold = float((hi + lo) / 2) if hi > lo else float(hi)
    actual = int(np.sum(widths >= threshold))
    return threshold, {
        "eligible_val": int(len(widths)),
        "target_fallback_count_val": k,
        "calibrated_fallback_count_val": actual,
        "calibrated_fallback_pct_val": 100.0 * actual / len(rec_val),
        "cutoff_boundary_tied": actual != k,
    }


def fallback_from_square(rec: pd.DataFrame, square_only: pd.DataFrame,
                         chosen: np.ndarray, min_face_px: float) -> pd.DataFrame:
    """Reuse exactly the original face-aware square decisions."""
    chosen = np.asarray(chosen, dtype=bool)
    if chosen.shape != (len(rec),):
        raise ValueError("chosen mask has wrong shape")
    valid = fallback_eligible(rec, square_only, min_face_px)
    if np.any(chosen & ~valid):
        raise ValueError("fallback selected an ineligible pair")
    out = square_only.copy(deep=True)
    out.loc[chosen, "mode"] = 2
    out.loc[chosen, "outcome"] = BOTH
    out.loc[chosen, "gap_contribution"] = 0.0
    out.loc[chosen, "square_quality"] = np.nan
    out.loc[chosen, "min_displayed_face_px"] = rec.loc[chosen, "full_min_face_px"].to_numpy(dtype=float)
    out.loc[chosen, "square_quality_loss"] = np.nan
    return out


def online_policy(rec: pd.DataFrame, delta: float, threshold: float,
                  min_face_px: float) -> pd.DataFrame:
    sq = apply_policy(rec, delta, 0.0, min_face_px)
    valid = fallback_eligible(rec, sq, min_face_px)
    chosen = valid & (rec["full_min_face_px"].to_numpy(dtype=float) >= threshold)
    return fallback_from_square(rec, sq, chosen, min_face_px)


def random_policy(rec: pd.DataFrame, square_only: pd.DataFrame, k: int,
                  min_face_px: float, seed: int) -> pd.DataFrame:
    valid = fallback_eligible(rec, square_only, min_face_px)
    eligible_idx = np.flatnonzero(valid)
    if k < 0 or k > len(eligible_idx):
        raise ValueError("random fallback count exceeds eligibility pool")
    rng = np.random.default_rng(seed)
    chosen = np.zeros(len(rec), dtype=bool)
    if k:
        chosen[rng.choice(eligible_idx, size=k, replace=False)] = True
    return fallback_from_square(rec, square_only, chosen, min_face_px)


def argmax_policy(rec: pd.DataFrame, square_only: pd.DataFrame) -> pd.DataFrame:
    out = square_only.copy(deep=True)
    out["mode"] = 0
    out["outcome"] = rec["arg_outcome"].to_numpy()
    out["gap_contribution"] = out["arg_gap_contribution"].to_numpy()
    out["square_quality"] = rec["arg_quality"].to_numpy()
    out["square_quality_loss"] = 0.0
    out["min_displayed_face_px"] = rec["arg_min_face_px"].to_numpy()
    return out


def _row(method: str, pairs: pd.DataFrame, rec: pd.DataFrame, out: pd.DataFrame,
         delta: float, budget: float, min_face_px: float) -> dict:
    return {"method": method, **summarize(pairs, rec, out, delta, budget, min_face_px)}


def _random_summary(draws: pd.DataFrame) -> pd.DataFrame:
    cols = (
        "gender_gap_pp", "race_matched_gap_pp", "max_six_abs_gap_pp",
        "fallback_pct", "total_change_pct", "mean_min_displayed_face_px",
        "two_faces_at_least_threshold_pct", "mean_saliency_retained_crops_only_pct",
    )
    rows = []
    for col in cols:
        x = draws[col].to_numpy(dtype=float)
        rows.append({"metric": col, "mean": x.mean(), "sd": x.std(ddof=1) if len(x)>1 else 0.0,
                     "q025": np.quantile(x, .025), "q975": np.quantile(x, .975)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=.03)
    ap.add_argument("--fallback-budget", type=float, default=.10)
    ap.add_argument("--min-face-px", type=float, default=32.0)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1701)
    ap.add_argument("--bootstrap-iters", type=int, default=0)
    ap.add_argument("--limit-per-comp", type=int, default=None,
                    help="Smoke test; not for research reporting")
    args = ap.parse_args()
    if args.draws < 2 or args.bootstrap_iters < 0:
        ap.error("--draws must be >=2 and --bootstrap-iters must be nonnegative")
    if args.limit_per_comp is not None and args.limit_per_comp <= 0:
        ap.error("--limit-per-comp must be positive")
    if not (0 <= args.delta < 1 and 0 <= args.fallback_budget <= 1 and args.min_face_px >= 0):
        ap.error("Require 0 <= delta < 1, 0 <= budget <= 1, min-face-px >= 0")

    print("Loading VALIDATION for threshold calibration ...", flush=True)
    pairs_val, rec_val = load_records("val", args.limit_per_comp)
    threshold, calibration = calibrate_threshold(rec_val, args.delta,
                                                   args.fallback_budget, args.min_face_px)
    print("Validation calibration:", {"threshold_px": threshold, **calibration})
    print("Loading TEST for frozen-threshold evaluation ...", flush=True)
    pairs, rec = load_records("test", args.limit_per_comp)
    sq = apply_policy(rec, args.delta, 0, args.min_face_px)
    online = online_policy(rec, args.delta, threshold, args.min_face_px)
    original = argmax_policy(rec, sq)
    offline = apply_policy(rec, args.delta, args.fallback_budget, args.min_face_px)
    k = int(np.sum(online["mode"].to_numpy() == 2))
    eligible_count = int(np.sum(fallback_eligible(rec, sq, args.min_face_px)))

    rows = [
        _row("argmax", pairs, rec, original, args.delta, args.fallback_budget, args.min_face_px),
        _row("face_aware_square_only", pairs, rec, sq, args.delta, args.fallback_budget, args.min_face_px),
        _row("offline_top_face_width_10pct", pairs, rec, offline, args.delta, args.fallback_budget, args.min_face_px),
        _row("validation_calibrated_online", pairs, rec, online, args.delta, args.fallback_budget, args.min_face_px),
    ]
    random_rows = []
    for j in range(args.draws):
        rand = random_policy(rec, sq, k, args.min_face_px, args.seed + j)
        random_rows.append({"draw": j, "seed": args.seed + j,
                            **summarize(pairs, rec, rand, args.delta, args.fallback_budget,
                                        args.min_face_px)})
    random_df = pd.DataFrame(random_rows)
    rows.append({"method": "random_eligible_fallback_mean", **{
        c: random_df[c].mean() for c in random_df.columns
        if c not in ("draw", "seed", "split") and pd.api.types.is_numeric_dtype(random_df[c])},
        "split": "test", "pairs": len(pairs)})

    dest = C.RESULTS / "face_aware_followup"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / "methods_test.csv", index=False)
    random_df.to_csv(dest / "random_fallback_draws.csv", index=False)
    _random_summary(random_df).to_csv(dest / "random_fallback_distribution.csv", index=False)
    per_comparison(pairs, online).to_csv(dest / "comparison_online_test.csv", index=False)
    pd.DataFrame([{"threshold_px":threshold,"online_fallback_test_count":k,
                   "eligible_fallback_test_count":eligible_count,
                   "online_fallback_test_pct":100*k/len(rec),
                   "random_draws":args.draws,"seed_start":args.seed,
                   **calibration}]).to_csv(dest / "calibration.csv", index=False)

    if args.bootstrap_iters:
        if args.limit_per_comp:
            print("Skipping bootstrap on small smoke test")
        elif args.bootstrap_iters < 50:
            ap.error("--bootstrap-iters must be >=50 when enabled")
        else:
            bootstrap(pairs, online, args.bootstrap_iters).to_csv(dest / "bootstrap_online_test.csv", index=False)
    print("\nMain results (percentage points unless otherwise stated):")
    print(pd.DataFrame(rows)[["method", "gender_gap_pp", "race_matched_gap_pp",
                              "max_six_abs_gap_pp", "fallback_pct", "total_change_pct",
                              "mean_min_displayed_face_px"]].to_string(index=False))
    print("Random control uses the SAME eligible pool and SAME fallback count as online policy.")
    print("It isolates the fallback assignment, NOT the entire face-aware algorithm.")
    print("These are secondary test analyses, not fresh holdout confirmation.")
    print(f"\nSaved aggregate files to {dest}")


if __name__ == "__main__":
    main()
