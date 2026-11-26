"""Evaluate the separate face-preserving square-preview policy.

Run from the repo root:
    python -m scripts.face_preserving_eval --split test --render 24

This does not call the existing rule dispatcher or replace its saved results.
There are no trainable parameters and no training stage.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cropbias import config as C
from cropbias.composites import composite_path
from cropbias.face_preserving import choose, render
from cropbias.geometry import crop_width
from cropbias.rules import View
from cropbias.stats import composite_weights, person_weights
from cropbias.twitter_saliency import load_store


def input_paths(split: str) -> tuple[Path, Path, Path]:
    return (C.DATA / "dataset.csv", C.DATA / "pairs.csv",
            C.WORK / "composites" / f"layout_{split}.csv")


def evaluate_one(pid: str, lay, sal, dg_cols, preview_path: Path | None) -> dict:
    W, H = int(lay.W), int(lay.H)
    v = View(W, H, crop_width(H, W, 1.0),
             (float(lay.f1x0), float(lay.f1x1)),
             (float(lay.f2x0), float(lay.f2x1)),
             sal.xs, sal.S, int(sal.salient[0]), dg_cols)
    choice = choose(v)
    if preview_path is not None:
        render(composite_path(pid, pid.split("_", 1)[0]), choice, preview_path)
    return {
        "pair_id": pid,
        "arg_outcome": v.arg["out"],
        "arg_gap_contribution": 0.5 * (int(v.arg["out"] == 1) - int(v.arg["out"] == 2)),
        "arg_saliency_retained": v.arg["tw"],
        "arg_deepgaze_retained": v.arg["dg"],
        "arg_face_cut": v.arg["cut"],
        "policy_outcome": choice.outcome,
        "policy_gap_contribution": 0.5 * (int(choice.outcome == 1) - int(choice.outcome == 2)),
        "fallback": int(choice.fallback),
        "square_feasible": int(choice.feasible),
        "x0": choice.x0,
        "x1": choice.x1,
        "face1_coverage": choice.face1_coverage,
        "face2_coverage": choice.face2_coverage,
        "policy_saliency_retained": choice.saliency_retained,
        "policy_deepgaze_retained": choice.deepgaze_retained,
        "preview_scale": choice.scale,
        "preview_padding_fraction": choice.padding_fraction,
    }


def interval(draws: np.ndarray) -> tuple[float, float]:
    good = draws[np.isfinite(draws)]
    if not len(good):
        return float("nan"), float("nan")
    return tuple(np.percentile(good, (2.5, 97.5)))


def aggregate(df: pd.DataFrame, pairs: pd.DataFrame, dataset: pd.DataFrame,
              split: str, iterations: int) -> pd.DataFrame:
    """Pair-aligned two-way person bootstrap, matching the repo's uncertainty unit."""
    people = {g: np.array(sorted(dataset.loc[
        (dataset.group == g) & (dataset.split == split), "qid"])) for g in C.GROUPS}
    if any(len(ids) == 0 for ids in people.values()):
        raise ValueError("no people in at least one required subgroup")
    counts = person_weights(people, iterations, C.BOOTSTRAP_SEED)
    weights = composite_weights(pairs, people, counts)
    rows = []
    for group, idx in pairs.groupby("comparison", sort=False).indices.items():
        p = df.iloc[idx]
        w = weights[:, idx].astype(np.float64)
        den = w.sum(axis=1)
        den[den == 0] = np.nan  # possible in a tiny --limit pilot
        def stat(col: str):
            y = p[col].to_numpy(dtype=np.float64)
            estimate = float(np.nanmean(y))
            draws = (w @ y) / den
            low, high = interval(draws)
            return estimate, low, high

        base = stat("arg_gap_contribution")
        new = stat("policy_gap_contribution")
        fallback = stat("fallback")
        sal_base = stat("arg_saliency_retained")
        sal_new = stat("policy_saliency_retained")
        rows.append({
            "comparison": group, "pairs": len(p),
            "argmax_gap_points": 100 * base[0],
            "argmax_gap_low": 100 * base[1], "argmax_gap_high": 100 * base[2],
            "policy_gap_points": 100 * new[0],
            "policy_gap_low": 100 * new[1], "policy_gap_high": 100 * new[2],
            "fallback_pct": 100 * fallback[0],
            "fallback_low": 100 * fallback[1], "fallback_high": 100 * fallback[2],
            "square_feasible_pct": 100 * p.square_feasible.mean(),
            "argmax_only_one_pct": 100 * p.arg_outcome.isin([1, 2]).mean(),
            "argmax_both_pct": 100 * (p.arg_outcome == 3).mean(),
            "argmax_no_face_pct": 100 * (p.arg_outcome == 0).mean(),
            "argmax_face_cut_pct": 100 * p.arg_face_cut.mean(),
            "argmax_saliency_retained_pct": 100 * sal_base[0],
            "policy_saliency_retained_pct": 100 * sal_new[0],
            "mean_preview_scale": p.preview_scale.mean(),
            "mean_padding_pct": 100 * p.preview_padding_fraction.mean(),
            "argmax_deepgaze_retained_pct": 100 * p.arg_deepgaze_retained.mean(),
            "policy_deepgaze_retained_pct": 100 * p.policy_deepgaze_retained.mean(),
            "deepgaze_pairs": int(p.policy_deepgaze_retained.notna().sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--limit", type=int, help="first N pair indices per comparison, for a pilot")
    ap.add_argument("--render", type=int, default=0, metavar="N",
                    help="save preview images for first N pairs (0=none, -1=all)")
    ap.add_argument("--bootstrap-iters", type=int, default=C.BOOTSTRAP_ITERS)
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if args.bootstrap_iters < 50:
        ap.error("--bootstrap-iters must be at least 50")
    if args.render < -1:
        ap.error("--render must be -1 or nonnegative")
    for p in (*input_paths(args.split), C.WORK / "saliency" / args.split / "orig.pkl"):
        if not p.is_file():
            ap.error(f"missing {p}; run the data/composites/saliency pipeline first")

    dataset = pd.read_csv(C.DATA / "dataset.csv")
    pairs = pd.read_csv(C.DATA / "pairs.csv")
    pairs = pairs.loc[pairs.split == args.split].copy()
    if args.limit is not None:
        pairs = pairs.loc[pairs.pair_id.str.rsplit("_", n=1).str[-1].astype(int) < args.limit]
    if pairs.empty:
        ap.error("no pairs found for requested split/limit")
    pairs = pairs.set_index("pair_id", drop=False)
    layout = pd.read_csv(input_paths(args.split)[2], index_col="pair_id")
    missing_layout = pairs.index.difference(layout.index)
    if len(missing_layout):
        ap.error(f"missing layouts for {len(missing_layout)} requested pairs")
    saliency = load_store(args.split, "orig")
    missing_saliency = set(pairs.index) - saliency.keys()
    if missing_saliency:
        ap.error(f"missing original saliency scores for {len(missing_saliency)} requested pairs")

    dg_path = C.WORK / "deepgaze" / "test.pkl"
    dg = pickle.loads(dg_path.read_bytes()) if args.split == "test" and dg_path.exists() else {}
    out_dir = C.WORK / "eval_face_preserving" / args.split
    rows = []
    for n, pid in enumerate(pairs.index):
        if n and n % 1000 == 0:
            print(f"face-preserving {args.split}: {n}/{len(pairs)} pairs", flush=True)
        show = args.render < 0 or n < args.render
        preview = out_dir / "previews" / f"{pid}.jpg" if show else None
        rows.append(evaluate_one(pid, layout.loc[pid], saliency[pid], dg.get(pid), preview))
    result = pd.DataFrame(rows).set_index("pair_id", drop=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "per_pair.csv", index=False)  # private pair-level details

    summary = aggregate(result.reset_index(drop=True), pairs.reset_index(drop=True),
                        dataset, args.split, args.bootstrap_iters)
    report_dir = C.RESULTS / "face_preserving"
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.split}" + (f"_pilot{args.limit}" if args.limit is not None else "")
    report = report_dir / f"summary{suffix}.csv"
    summary.to_csv(report, index=False)
    print(summary[["comparison", "argmax_gap_points", "policy_gap_points",
                   "fallback_pct", "square_feasible_pct",
                   "argmax_saliency_retained_pct", "policy_saliency_retained_pct"]].to_string(index=False))
    print(f"Saved {report} and {out_dir / 'per_pair.csv'}")
    print("The policy gap is zero by design conditional on the detected face boxes;")
    print("saliency/DeepGaze retained on fallback is 100% of original image mass, not a readability score.")


if __name__ == "__main__":
    main()
