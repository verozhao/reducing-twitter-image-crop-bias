"""Tables behind the final report that the main analysis does not write.

  python -m scripts.report_tables      # writes results/report_tables/*.csv

bootstrap_seed_check.csv  Interval endpoints of the argmax gaps and of the share of the pooled
                          gap each rule removes, with the people-level bootstrap seeded 8 to 12
                          instead of 7, and their largest move from seed 7.
fixes_table.csv           Every single-crop fix on the test pairs: gender gap (mean of the four
                          cross-gender comparisons) and race gap (mean of the two same-gender
                          comparisons) with people-level intervals, the share of crops changed in
                          those comparisons, and the gap a random switch of the same share of
                          crops in each comparison would leave.
rules_ten_draws.csv       Share of the pooled gap each rule removes: main draw, and mean, lowest
                          and highest over the ten draws of the test pairs.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from cropbias import config as C
from cropbias.analysis import COMP_NAMES, Boot, Split, gap_table, orientation, pooled_oriented
from cropbias.stats import composite_weights, person_weights, summarize

OUT = C.RESULTS / "report_tables"
CROSS_GENDER = ["BF-BM", "WF-WM", "BF-WM", "WF-BM"]   # the woman is group 1 in each
SAME_GENDER = ["WF-BF", "WM-BM"]                       # the White person is group 1 in each
TABLE2_RULES = ["tie_rate20", "sampling_T0.3", "averaging_n2", "stability_geom_m1", "stability_main_m1"]
TABLE3_RULES = ["tie_rate40", "sampling_T1"]
SCORE_FIXES = {
    "grayscale input": C.WORK / "map_correction" / "preprocess_eval" / "test" / "gray",
    "matched brightness and contrast": C.WORK / "map_correction" / "preprocess_eval" / "test" / "matched_luminance",
    "learned correction, run 1": C.WORK / "map_correction" / "evaluation" / "test",
    "learned correction, run 2": C.WORK / "map_correction" / "evaluation_seed44" / "test",
    "learned correction, run 3": C.WORK / "map_correction" / "evaluation_seed45" / "test",
    "equalized group scores (oracle, uses labels)": C.WORK / "score_fixes" / "equalized" / "test",
}


def reseeded(bt: Boot, test: Split, seed: int) -> Boot:
    b = copy.copy(bt)
    counts = person_weights(bt.people, C.BOOTSTRAP_ITERS, seed)
    b.W = composite_weights(test.pairs, bt.people, counts).astype(np.float64)
    return b


def removed_share(test: Split, bt: Boot, Y: np.ndarray, sign: dict, cfgs: list[str]) -> dict:
    G_pt, G_b = pooled_oriented(bt.comp_stats(Y), sign)
    a = test.ci["argmax"]
    out = {}
    for c in cfgs:
        j = test.ci[c]
        out[c] = summarize((G_pt[a] - G_pt[j]) / G_pt[a], (G_b[:, a] - G_b[:, j]) / G_b[:, a])
    return out


def seed_check(test: Split, bt: Boot, Y: np.ndarray, sign: dict) -> pd.DataFrame:
    rows = []
    for seed in [C.BOOTSTRAP_SEED, 8, 9, 10, 11, 12]:
        b = bt if seed == C.BOOTSTRAP_SEED else reseeded(bt, test, seed)
        g = gap_table(test, b, Y[:, [test.ci["argmax"]]], ["argmax"])
        rows += [{"seed": seed, "quantity": f"argmax gap {r.comparison}", "lo": r.lo, "hi": r.hi} for r in g.itertuples()]
        for c, s in removed_share(test, b, Y, sign, TABLE2_RULES).items():
            rows.append({"seed": seed, "quantity": f"{c} share of pooled gap removed", "lo": s["lo"], "hi": s["hi"]})
    df = pd.DataFrame(rows)
    base = df[df.seed == C.BOOTSTRAP_SEED].set_index("quantity")[["lo", "hi"]]
    df["move_from_seed_7"] = [max(abs(r.lo - base.loc[r.quantity, "lo"]), abs(r.hi - base.loc[r.quantity, "hi"]))
                              for r in df.itertuples()]
    return df


def summary_rows(name: str, base: np.ndarray, new: np.ndarray, changed: np.ndarray, comp: np.ndarray,
                 W: np.ndarray, labels: str) -> dict:
    """Gender and race gaps with intervals, changed share, and the random-switch benchmark."""
    row = {"fix": name, "labels_at_crop_time": labels}
    for label, comps in (("gender", CROSS_GENDER), ("race", SAME_GENDER)):
        masks = [comp == c for c in comps]
        pt = np.mean([new[m].mean() for m in masks])
        bt = np.mean([(W[:, m] @ new[m]) / W[:, m].sum(axis=1) for m in masks], axis=0)
        s = summarize(pt, bt)
        x = np.array([changed[m].mean() for m in masks])
        g0 = np.array([base[m].mean() for m in masks])
        row.update({f"{label}_gap": s["est"], f"{label}_lo": s["lo"], f"{label}_hi": s["hi"],
                    f"{label}_changed": x.mean(), f"{label}_random_switch": float(np.mean(g0 * (1 - 2 * x)))})
    return row


def fixes_table(test: Split, bt: Boot, Y: np.ndarray) -> pd.DataFrame:
    comp = test.comp
    a = test.ci["argmax"]
    change = test.metric("square", "change")
    rows = [summary_rows("argmax (Twitter)", Y[:, a], Y[:, a], np.zeros(len(Y)), comp, bt.W, "no")]
    for c in TABLE3_RULES:
        rows.append(summary_rows(c, Y[:, a], Y[:, test.ci[c]], change[:, test.ci[c]], comp, bt.W, "no"))
    order = pd.Index(test.desc.index)
    for name, d in SCORE_FIXES.items():
        f = d / "per_pair.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).set_index("pair_id").loc[order]
        changed = (df.baseline_outcome.to_numpy() != df.new_outcome.to_numpy()).astype(float)
        labels = "yes" if "uses labels" in name else "no"
        rows.append(summary_rows(name, df.baseline_gap.to_numpy(float), df.new_gap.to_numpy(float), changed,
                                 comp, bt.W, labels))
    return pd.DataFrame(rows)


def ten_draws() -> pd.DataFrame:
    r = pd.read_csv(C.RESULTS / "seeds" / "seed_results.csv")
    rows = []
    for c in ["tie_rate20", "tie_rate40", "sampling_T0.3", "sampling_T1", "averaging_n2"]:
        s = r[r.metric == f"{c} gap removed"]
        main = s[s.design.str.startswith("main")].est.iloc[0]
        rows.append({"rule": c, "main_draw": main, "mean": s.est.mean(), "lowest": s.est.min(), "highest": s.est.max(),
                     "draws": len(s)})
    return pd.DataFrame(rows)


def main() -> None:
    test, val = Split("test"), Split("val")
    bt = Boot(test)
    Y = test.y_gap("square")
    sign = orientation(val, "square")
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = seed_check(test, bt, Y, sign)
    seeds.to_csv(OUT / "bootstrap_seed_check.csv", index=False)
    moved = seeds.assign(kind=np.where(seeds.quantity.str.startswith("argmax"), "argmax gaps", "removed shares"))
    print("largest move of an interval endpoint from seed 7 (points):")
    print((100 * moved.groupby("kind").move_from_seed_7.max()).round(2).to_string())
    fixes = fixes_table(test, bt, Y)
    fixes.to_csv(OUT / "fixes_table.csv", index=False)
    print((fixes.set_index("fix").select_dtypes("number") * 100).round(1).to_string())
    draws = ten_draws()
    draws.to_csv(OUT / "rules_ten_draws.csv", index=False)
    print((draws.set_index("rule").drop(columns="draws") * 100).round(1).to_string())


if __name__ == "__main__":
    main()
