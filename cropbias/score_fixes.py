"""Score fixes on one footing: an upper bound and a common summary.

Upper bound. The gender gap comes from a steady shift in face scores. If a correction
removed the average score difference between the four groups, how much of each gap
would be left? We estimate one score effect per group from the face-score margins of
validation composites, subtract it from each photo's half of the map, and crop at the new
highest point. This uses demographic labels at crop time, which a deployed system should
not do, so it is a bound for score corrections and not a method.

Summary. Every fix is scored on the same test pairs against the same argmax crops, with
the people-level bootstrap of the main analysis, so differences between fixes are paired.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .map_correction import crop_stats
from .stats import composite_weights, person_weights, summarize
from .twitter_saliency import load_store

COMPS = [f"{a}-{b}" for a, b in C.COMPARISONS]
CROSS_GENDER = ["BF-BM", "WF-WM", "BF-WM", "WF-BM"]   # the woman is group 1 in each
WITHIN_GENDER = ["WF-BF", "WM-BM"]                     # the White person is group 1 in each
GENDER = "gender (women - men), mean of the four cross-gender comparisons"
RACE = "race (White - Black), mean of the two same-gender comparisons"
LARGEST = "largest absolute gap of the six comparisons"


# ----------------------------------------------------------------------------
# Upper bound: equalized group scores
# ----------------------------------------------------------------------------
def offsets_from_margins(g1, g2, margin) -> dict[str, float]:
    """Least-squares group effects with margin = effect[g1] - effect[g2] + noise.
    Margins identify the effects only up to a constant; the minimum-norm solution centers them."""
    groups = list(C.GROUPS)
    X = np.zeros((len(margin), len(groups)))
    rows = np.arange(len(margin))
    X[rows, [groups.index(g) for g in g1]] += 1.0
    X[rows, [groups.index(g) for g in g2]] -= 1.0
    eff, *_ = np.linalg.lstsq(X, np.asarray(margin, float), rcond=None)
    return dict(zip(groups, eff))


def group_offsets(split: str = "val") -> dict[str, float]:
    desc = pd.read_csv(C.WORK / "eval" / f"{split}_desc.csv", index_col=0)
    pairs = pd.read_csv(C.DATA / "pairs.csv").set_index("pair_id").loc[desc.index]
    return offsets_from_margins(pairs.g1, pairs.g2, (desc.s1_face - desc.s2_face).to_numpy())


def column_shift(xs: np.ndarray, w_left: float, left_group: str, right_group: str,
                 offsets: dict[str, float]) -> np.ndarray:
    """Per-column amount to subtract: each photo's half of the map loses its group's effect."""
    return np.where(np.asarray(xs) < w_left, offsets[left_group], offsets[right_group])


def equalized_per_pair(split: str = "test", offsets: dict[str, float] | None = None) -> pd.DataFrame:
    offsets = offsets or group_offsets("val")
    pairs = pd.read_csv(C.DATA / "pairs.csv").query("split == @split")
    lay = pd.read_csv(C.WORK / "composites" / f"layout_{split}.csv", index_col="pair_id")
    store = load_store(split, "orig")
    rows = []
    for r in pairs.itertuples():
        L, s = lay.loc[r.pair_id], store[r.pair_id]
        left, right = (r.g1, r.g2) if bool(r.g1_left) else (r.g2, r.g1)
        shifted = s.S - column_shift(s.xs, float(L.w_left), left, right, offsets)[None, :]
        before, after = crop_stats(L, s), crop_stats(L, s, shifted)
        rows.append({"pair_id": r.pair_id, "baseline_outcome": before["outcome"],
                     "new_outcome": after["outcome"], "baseline_gap": before["gap"],
                     "new_gap": after["gap"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Common summary
# ----------------------------------------------------------------------------
def method_rows(name: str, df: pd.DataFrame, comp: np.ndarray, W: np.ndarray) -> list[dict]:
    """Gaps under argmax and under the fix, and their paired change, per comparison and pooled.
    df rows must be in the same order as comp and the columns of W."""
    masks = [comp == c for c in COMPS]

    def per_comp(v):
        pt = np.array([v[m].mean() for m in masks])
        bt = np.stack([(W[:, m] @ v[m]) / W[:, m].sum(axis=1) for m in masks], axis=1)
        return pt, bt

    bp, bb = per_comp(df.baseline_gap.to_numpy(float))
    fp, fb = per_comp(df.new_gap.to_numpy(float))
    rows = []

    def add(measure, a_pt, a_bt, f_pt, f_bt):
        for value, pt, bt in (("argmax", a_pt, a_bt), ("fixed", f_pt, f_bt), ("change", f_pt - a_pt, f_bt - a_bt)):
            rows.append({"method": name, "measure": measure, "value": value, **summarize(pt, bt)})

    for k, c in enumerate(COMPS):
        add(c, bp[k], bb[:, k], fp[k], fb[:, k])
    for label, cs in ((GENDER, CROSS_GENDER), (RACE, WITHIN_GENDER)):
        i = [COMPS.index(c) for c in cs]
        add(label, bp[i].mean(), bb[:, i].mean(axis=1), fp[i].mean(), fb[:, i].mean(axis=1))
    add(LARGEST, np.abs(bp).max(), np.abs(bb).max(axis=1), np.abs(fp).max(), np.abs(fb).max(axis=1))
    changed = (df.baseline_outcome.to_numpy() != df.new_outcome.to_numpy()).mean()
    no_face = (df.new_outcome.to_numpy() == 0).mean()
    rows.append({"method": name, "measure": "share of crops changed", "value": "fixed", "est": changed})
    rows.append({"method": name, "measure": "share of crops with no face", "value": "fixed", "est": no_face})
    return rows


def summarize_methods(per_pair: dict[str, pd.DataFrame], split: str = "test",
                      iters: int = C.BOOTSTRAP_ITERS) -> pd.DataFrame:
    pairs = pd.read_csv(C.DATA / "pairs.csv").query("split == @split").reset_index(drop=True)
    ds = pd.read_csv(C.DATA / "dataset.csv")
    people = {g: np.array(sorted(ds.qid[(ds.group == g) & (ds.split == split)])) for g in C.GROUPS}
    counts = person_weights(people, iters, C.BOOTSTRAP_SEED)
    W = composite_weights(pairs, people, counts).astype(np.float64)
    rows = []
    for name, df in per_pair.items():
        df = df.set_index("pair_id").loc[pairs.pair_id].reset_index()
        rows += method_rows(name, df, pairs.comparison.to_numpy(), W)
    return pd.DataFrame(rows)
