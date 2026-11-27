"""Offline square-preview fairness/quality trade-off experiment.

From repository root:
    python -m scripts.budgeted_face_aware --split val --sweep
    python -m scripts.budgeted_face_aware --split test --delta 0.03 \
        --fallback-budget 0.10 --bootstrap-iters 500

Reads PRIVATE original WikiCeleb pairs/layout/Twitter saliency caches. Does not
run the Twitter binary, load photos, train a model, or overwrite prior results.

Methods:
  1. Argmax: unchanged Twitter crop.
  2. Face-aware: if argmax does not retain BOTH faces, use the best square
     retaining BOTH only when its saliency mass >= (1-delta) * argmax mass.
  3. Budgeted full-image fallback: among remaining non-BOTH images, letterbox
     up to floor(budget * total images) images whose *minimum* face width after
     scaling into the square is >= min_face_px. Rank by that minimum width,
     ties broken by pair_id. This is an OFFLINE BATCH policy, not a per-image
     online policy, and never uses gender/race labels for selection.

The preview face-size metric is an image-geometry proxy, not a user study.
Whole-image fallback is NOT scored as a high-quality crop just because all
saliency is retained. Its saliency-retention field is missing by design.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from cropbias import config as C
from cropbias.face_preserving import choose
from cropbias.geometry import BOTH, crop_width, window
from cropbias.rules import View
from cropbias.stats import composite_weights, person_weights
from cropbias.twitter_saliency import load_store


COMPS = [f"{a}-{b}" for a, b in C.COMPARISONS]


def _face_width_in_crop(face: tuple[float, float], x0: int, x1: int) -> float:
    return float(max(0.0, min(face[1], x1) - max(face[0], x0)))


def _record(pid: str, layout, sal) -> dict:
    """Prepare all individual-image alternatives once; no demographic inputs."""
    W, H = int(layout.W), int(layout.H)
    # A handful of validation composites are 1-2 px narrower than height.
    # They have no horizontal square window; treat their Twitter crop as
    # width-W for horizontal face coverage and leave it unchanged.
    if sal.S.ndim != 2 or sal.S.shape[1] != len(sal.xs):
        raise ValueError(f"{pid}: malformed saliency map")
    if not np.isfinite(sal.S).all() or np.any(sal.xs < 0) or np.any(sal.xs >= W):
        raise ValueError(f"{pid}: nonfinite/out-of-bounds saliency map")
    f1 = (float(layout.f1x0), float(layout.f1x1))
    f2 = (float(layout.f2x0), float(layout.f2x1))
    if not (0 <= f1[0] < f1[1] <= W and 0 <= f2[0] < f2[1] <= W):
        raise ValueError(f"{pid}: invalid face x coordinates")
    v = View(W, H, crop_width(H, W, 1.0), f1, f2,
             sal.xs, sal.S, int(sal.salient[0]))
    if v.crop_w != min(H, W):
        raise ValueError(f"{pid}: unexpected crop width")
    a0, a1 = window(v.salient_x, v.crop_w, W)
    a0, a1 = int(a0), int(a1)
    alt = choose(v) if W >= H else None  # exact square window, when possible
    orig_out = int(v.arg["out"])
    arg_quality = float(v.arg["tw"])
    best_quality = float(alt.saliency_retained) if alt is not None and alt.feasible else float("nan")
    min_arg_width = min(_face_width_in_crop(f1, a0, a1),
                        _face_width_in_crop(f2, a0, a1))
    min_best_width = (min(_face_width_in_crop(f1, int(alt.x0), int(alt.x1)),
                          _face_width_in_crop(f2, int(alt.x0), int(alt.x1)))
                      if alt is not None and alt.feasible else float("nan"))
    full_scale = min(1.0, H / W)
    min_full_width = min(f1[1] - f1[0], f2[1] - f2[0]) * full_scale
    return {
        "pair_id": pid,
        "arg_outcome": orig_out,
        "arg_quality": arg_quality,
        "arg_x0": a0,
        "square_feasible": bool(alt is not None and alt.feasible),
        "best_both_quality": best_quality,
        "arg_min_face_px": min_arg_width,
        "best_both_min_face_px": min_best_width,
        "full_min_face_px": min_full_width,
        "full_scale": full_scale,
    }


def load_records(split: str, limit_per_comp: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs_path = C.DATA / "pairs.csv"
    dataset_path = C.DATA / "dataset.csv"
    layout_path = C.WORK / "composites" / f"layout_{split}.csv"
    sal_path = C.WORK / "saliency" / split / "orig.pkl"
    for p in (pairs_path, dataset_path, layout_path, sal_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing: {p}\nSee the setup commands in the instructions.")
    pairs = pd.read_csv(pairs_path)
    pairs = pairs.loc[pairs.split == split].copy()
    if limit_per_comp is not None:
        # Each pair ID ends in a zero-padded numeric index, created per comparison.
        suffix = pairs.pair_id.str.rsplit("_", n=1).str[-1].astype(int)
        pairs = pairs.loc[suffix < limit_per_comp].copy()
    if pairs.empty or pairs.pair_id.duplicated().any():
        raise ValueError("No pairs or duplicate pair IDs")
    if not set(pairs.comparison).issubset(COMPS):
        raise ValueError("Unexpected group comparisons")
    pairs = pairs.sort_values("pair_id").reset_index(drop=True)
    layout = pd.read_csv(layout_path).set_index("pair_id", verify_integrity=True)
    saliency = load_store(split, "orig")  # Only deserialize trusted group-shared pickle files.
    missing_layout = set(pairs.pair_id) - set(layout.index)
    missing_sal = set(pairs.pair_id) - set(saliency)
    if missing_layout or missing_sal:
        raise ValueError(f"Data mismatch: {len(missing_layout)} layouts and "
                         f"{len(missing_sal)} saliency entries missing. "
                         "Do not mix caches from different pair seeds.")
    # Additional compatibility check to catch common cross-seed mismatches.
    lside = layout.loc[pairs.pair_id, "g1_left"].to_numpy(dtype=bool)
    pside = pairs.g1_left.to_numpy(dtype=bool)
    if not np.array_equal(lside, pside):
        raise ValueError("pairs.csv and layout disagree on left/right positions; "
                         "these files may come from different seed runs")
    rows = []
    for i, pid in enumerate(pairs.pair_id):
        if i and i % 4000 == 0:
            print(f"Loaded {i}/{len(pairs)} composites", flush=True)
        rows.append(_record(pid, layout.loc[pid], saliency[pid]))
    rec = pd.DataFrame(rows)
    if not rec.pair_id.equals(pairs.pair_id):
        raise AssertionError("Pair ordering was lost")
    return pairs, rec


def apply_policy(rec: pd.DataFrame, delta: float,
                 fallback_budget: float, min_face_px: float) -> pd.DataFrame:
    """Decide only from geometry and saliency, never pair demographics."""
    if not (0 <= delta < 1 and 0 <= fallback_budget <= 1 and min_face_px >= 0):
        raise ValueError("Invalid delta, budget, or minimum face size")
    out = rec[["pair_id", "arg_outcome", "arg_quality", "square_feasible"]].copy()
    n = len(rec)
    base_out = rec.arg_outcome.to_numpy(dtype=np.int8)
    # Only replace originally exclusive/empty crops, so already-BOTH crops are stable.
    alt_ok = (rec.square_feasible.to_numpy(dtype=bool)
              & (base_out != BOTH)
              & (rec.best_both_quality.to_numpy(dtype=float)
                 >= (1.0 - delta) * rec.arg_quality.to_numpy(dtype=float) - 1e-12))
    mode = np.zeros(n, dtype=np.int8)  # 0=argmax; 1=both-square; 2=full-image fallback
    mode[alt_ok] = 1
    remaining = (mode == 0) & (base_out != BOTH)
    eligible = remaining & (rec.full_min_face_px.to_numpy(dtype=float) >= min_face_px)
    k = min(int(math.floor(fallback_budget * n + 1e-10)), int(eligible.sum()))
    if k:
        # Global, group-blind OFFLINE budget allocation; deterministic tie-break.
        indices = np.flatnonzero(eligible)
        eligible_rows = rec.iloc[indices]
        order = np.lexsort((eligible_rows.pair_id.to_numpy(),
                            -eligible_rows.full_min_face_px.to_numpy(dtype=float)))
        mode[indices[order[:k]]] = 2
    outcome = np.where(mode == 0, base_out, BOTH).astype(np.int8)
    quality = np.where(mode == 1, rec.best_both_quality.to_numpy(dtype=float),
                       rec.arg_quality.to_numpy(dtype=float))
    quality[mode == 2] = np.nan  # Deliberately do NOT call full image a quality-1 crop.
    min_displayed = np.where(mode == 1,
                             rec.best_both_min_face_px.to_numpy(dtype=float),
                             rec.arg_min_face_px.to_numpy(dtype=float))
    min_displayed = np.where(mode == 2, rec.full_min_face_px.to_numpy(dtype=float),
                             min_displayed)
    out["mode"] = mode
    out["outcome"] = outcome
    out["gap_contribution"] = 0.5 * ((outcome == 1).astype(float) - (outcome == 2).astype(float))
    out["arg_gap_contribution"] = 0.5 * ((base_out == 1).astype(float) - (base_out == 2).astype(float))
    out["square_quality"] = quality
    out["min_displayed_face_px"] = min_displayed
    out["square_quality_loss"] = np.where(mode == 1,
        np.maximum(0.0, rec.arg_quality.to_numpy(dtype=float)
                   - rec.best_both_quality.to_numpy(dtype=float)), 0.0)
    out.loc[mode == 2, "square_quality_loss"] = np.nan
    return out


def _gap_by_comp(contrib: np.ndarray, comparisons: np.ndarray) -> dict[str, float]:
    return {c: float(np.mean(contrib[comparisons == c])) for c in COMPS
            if np.any(comparisons == c)}


def _fairness(gaps: dict[str, float]) -> dict[str, float]:
    """In percentage points. Signed gap = P(g1 kept) - P(g2 kept)."""
    if not all(c in gaps for c in COMPS):
        return {"gender_gap_pp": float("nan"), "race_matched_gap_pp": float("nan"),
                "max_six_abs_gap_pp": float("nan")}
    return {
        "gender_gap_pp": 100 * (gaps["BF-BM"] + gaps["WF-WM"]) / 2,
        "race_matched_gap_pp": 100 * (gaps["WF-BF"] + gaps["WM-BM"]) / 2,
        "max_six_abs_gap_pp": 100 * max(abs(x) for x in gaps.values()),
    }


def summarize(pairs: pd.DataFrame, rec: pd.DataFrame, out: pd.DataFrame,
              delta: float, fallback_budget: float, min_face_px: float) -> dict:
    g = pairs.comparison.to_numpy()
    base = _gap_by_comp(out.arg_gap_contribution.to_numpy(dtype=float), g)
    current = _gap_by_comp(out.gap_contribution.to_numpy(dtype=float), g)
    a, b = _fairness(base), _fairness(current)
    mode = out["mode"].to_numpy()
    quality_non_fallback = out.loc[mode != 2, "square_quality"]
    changed_square = out.loc[mode == 1, "square_quality_loss"]
    row = {
        "split": str(pairs.split.iloc[0]), "pairs": len(pairs),
        "delta": delta, "fallback_budget": fallback_budget,
        "min_face_px_threshold": min_face_px,
        "arg_gender_gap_pp": a["gender_gap_pp"],
        "gender_gap_pp": b["gender_gap_pp"],
        "arg_race_matched_gap_pp": a["race_matched_gap_pp"],
        "race_matched_gap_pp": b["race_matched_gap_pp"],
        "arg_max_six_abs_gap_pp": a["max_six_abs_gap_pp"],
        "max_six_abs_gap_pp": b["max_six_abs_gap_pp"],
        "both_square_change_pct": 100 * np.mean(mode == 1),
        "fallback_pct": 100 * np.mean(mode == 2),
        "total_change_pct": 100 * np.mean(mode != 0),
        "both_face_pct": 100 * np.mean(out.outcome == BOTH),
        "two_faces_at_least_threshold_pct": 100 * np.mean(
            out.min_displayed_face_px.to_numpy(dtype=float) >= min_face_px),
        "mean_min_displayed_face_px": float(out.min_displayed_face_px.mean()),
        "mean_saliency_retained_crops_only_pct": 100 * float(quality_non_fallback.mean()),
        "mean_loss_on_changed_squares_pp": 100 * float(changed_square.mean())
            if len(changed_square) else 0.0,
        "square_feasible_pct": 100 * float(rec.square_feasible.mean()),
    }
    return row


def per_comparison(pairs: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"comparison": pairs.comparison.to_numpy(),
        "base": out.arg_gap_contribution.to_numpy(),
        "policy": out.gap_contribution.to_numpy(),
        "fallback": (out["mode"].to_numpy() == 2).astype(float),
        "both": (out.outcome.to_numpy() == BOTH).astype(float),
        "min_face_px": out.min_displayed_face_px.to_numpy()})
    d = df.groupby("comparison", sort=False).agg(
        pairs=("base", "size"), argmax_gap=("base", "mean"),
        policy_gap=("policy", "mean"), fallback_rate=("fallback", "mean"),
        both_rate=("both", "mean"), mean_min_face_px=("min_face_px", "mean"))
    for key in ("argmax_gap", "policy_gap", "fallback_rate", "both_rate"):
        d[key + "_pct_or_pp"] = 100 * d.pop(key)
    return d.reset_index()


def bootstrap(pairs: pd.DataFrame, out: pd.DataFrame, iters: int) -> pd.DataFrame:
    """Paired two-way people bootstrap for selected policy, NOT the entire sweep.

    The offline fallback assignment is held fixed in each bootstrap replicate.
    """
    dataset = pd.read_csv(C.DATA / "dataset.csv")
    split = str(pairs.split.iloc[0])
    people = {g: np.array(sorted(dataset.loc[
        (dataset.group == g) & (dataset.split == split), "qid"])) for g in C.GROUPS}
    if any(not len(v) for v in people.values()):
        raise ValueError("Missing demographic groups in person table")
    w = composite_weights(pairs, people,
                          person_weights(people, iters, C.BOOTSTRAP_SEED))
    comp = pairs.comparison.to_numpy()
    draws = {}
    for prefix, column in [("arg", "arg_gap_contribution"),
                           ("new", "gap_contribution")]:
        y = out[column].to_numpy(dtype=np.float64)
        one = {}
        for c in COMPS:
            mask = comp == c
            if not mask.any():
                raise ValueError("Bootstrapping requires all six comparisons")
            wc = w[:, mask].astype(np.float64)
            den = wc.sum(axis=1)
            one[c] = np.divide(wc @ y[mask], den,
                               out=np.full(iters, np.nan), where=den > 0)
        draws[prefix + "_gender_gap_pp"] = 50 * (one["BF-BM"] + one["WF-WM"])
        draws[prefix + "_race_matched_gap_pp"] = 50 * (one["WF-BF"] + one["WM-BM"])
        draws[prefix + "_max_six_abs_gap_pp"] = 100 * np.max(
            np.abs(np.stack([one[c] for c in COMPS])), axis=0)
    for col in ("gender_gap_pp", "race_matched_gap_pp", "max_six_abs_gap_pp"):
        draws["change_" + col] = draws["new_" + col] - draws["arg_" + col]
    rows = []
    for name, x in draws.items():
        x = x[np.isfinite(x)]
        rows.append({"metric": name,
            "ci_low_95": float(np.percentile(x, 2.5)) if len(x) else np.nan,
            "ci_high_95": float(np.percentile(x, 97.5)) if len(x) else np.nan,
            "valid_bootstrap_draws": len(x)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--delta", type=float, default=0.03,
                    help="max relative loss in argmax saliency mass for BOTH square (default 0.03)")
    ap.add_argument("--fallback-budget", type=float, default=0.10,
                    help="maximum fraction of all pairs allowed full-image display (default 0.10)")
    ap.add_argument("--min-face-px", type=float, default=32,
                    help="minimum width of EACH displayed face to permit fallback (default 32)")
    ap.add_argument("--sweep", action="store_true",
                    help="validation-only sweep over predeclared quality/fallback budgets")
    ap.add_argument("--bootstrap-iters", type=int, default=0,
                    help="people-level 95%% CI for selected policy (e.g. 500); default disabled")
    ap.add_argument("--limit-per-comp", type=int, default=None,
                    help="small local smoke test; never use pilot as final research result")
    ap.add_argument("--save-per-pair", action="store_true",
                    help="write PRIVATE pair-level results to work/, not public results/")
    args = ap.parse_args()
    if args.limit_per_comp is not None and args.limit_per_comp <= 0:
        ap.error("--limit-per-comp must be positive")
    if args.bootstrap_iters and args.bootstrap_iters < 50:
        ap.error("--bootstrap-iters must be 0 or >= 50")
    if args.sweep and args.split == "test":
        ap.error("Use validation to inspect the sweep; freeze parameters before test")
    if not (0 <= args.delta < 1 and 0 <= args.fallback_budget <= 1 and args.min_face_px >= 0):
        ap.error("Require 0 <= delta < 1, 0 <= fallback-budget <= 1, min-face-px >= 0")
    pairs, rec = load_records(args.split, args.limit_per_comp)
    result_dir = C.RESULTS / "budgeted_face_aware"
    result_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_pilot" if args.limit_per_comp else ""
    setting = (f"{args.split}_d{round(args.delta * 1000):03d}"
               f"_b{round(args.fallback_budget * 1000):03d}"
               f"_px{args.min_face_px:g}{suffix}")
    if args.sweep:
        rows = []
        for delta in (0.0, 0.01, 0.03, 0.05, 0.10):
            for fb in (0.0, 0.05, 0.10, 0.20):
                out = apply_policy(rec, delta, fb, args.min_face_px)
                rows.append(summarize(pairs, rec, out, delta, fb, args.min_face_px))
        path = result_dir / f"summary_{args.split}_sweep_px{args.min_face_px:g}{suffix}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(pd.DataFrame(rows)[["delta", "fallback_budget", "gender_gap_pp",
                                  "race_matched_gap_pp", "max_six_abs_gap_pp",
                                  "fallback_pct", "both_square_change_pct",
                                  "two_faces_at_least_threshold_pct"]].to_string(index=False))
        print(f"Saved {path}")
        return
    out = apply_policy(rec, args.delta, args.fallback_budget, args.min_face_px)
    row = summarize(pairs, rec, out, args.delta, args.fallback_budget, args.min_face_px)
    pd.DataFrame([row]).to_csv(result_dir / f"summary_{setting}.csv", index=False)
    per_comparison(pairs, out).to_csv(result_dir / f"comparison_{setting}.csv", index=False)
    print(pd.DataFrame([row]).T.to_string(header=False))
    if args.bootstrap_iters:
        if args.limit_per_comp:
            print("WARNING: pilot bootstrap intervals are not suitable for research claims.")
        ci = bootstrap(pairs, out, args.bootstrap_iters)
        ci.to_csv(result_dir / f"bootstrap_{setting}.csv", index=False)
        print("\nPaired people-level bootstrap 95% intervals (percentage points):")
        print(ci.to_string(index=False))
    if args.save_per_pair:
        private_dir = C.WORK / "budgeted_face_aware"
        private_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(private_dir / f"per_pair_{setting}.csv", index=False)
        print(f"Private per-pair file saved under {private_dir}")
    print(f"\nSaved summary and per-comparison CSV under {result_dir}")
    print("IMPORTANT: crop-only saliency retention excludes full-image fallback;")
    print("its cost is shown by fallback rate and displayed-face-size proxy.")


if __name__ == "__main__":
    main()
