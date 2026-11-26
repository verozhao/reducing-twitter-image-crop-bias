"""Score fixes: the equalized-score bound, the choice of correction setting, one summary.

  python -m scripts.score_fixes bound     # upper bound from equalized group scores (uses labels)
  python -m scripts.score_fixes select    # choose the correction setting on validation people only
  python -m scripts.score_fixes summary   # results/score_fixes/summary_test.csv

scripts/run_score_fixes.sh runs every step in order.
"""
from __future__ import annotations

import argparse

import pandas as pd

from cropbias import config as C
from cropbias.score_fixes import COMPS, GENDER, LARGEST, RACE, equalized_per_pair, group_offsets, summarize_methods

GRID = C.WORK / "map_correction" / "grid"
OUT = C.RESULTS / "score_fixes"
SOURCES = {   # method -> per-pair outcomes on the test pairs, written by each run
    "learned correction, original objective": C.WORK / "map_correction" / "evaluation" / "test",
    "learned correction, setting chosen on validation": C.WORK / "map_correction" / "evaluation_selected" / "test",
    "grayscale input": C.WORK / "map_correction" / "preprocess_eval" / "test" / "gray",
    "matched brightness and contrast": C.WORK / "map_correction" / "preprocess_eval" / "test" / "matched_luminance",
    "equalized group scores (uses labels, bound)": C.WORK / "score_fixes" / "equalized" / "test",
}


def bound() -> None:
    offsets = group_offsets("val")
    target = SOURCES["equalized group scores (uses labels, bound)"]
    target.mkdir(parents=True, exist_ok=True)
    equalized_per_pair("test", offsets).to_csv(target / "per_pair.csv", index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.Series(offsets, name="score_effect").rename_axis("group").to_csv(OUT / "group_score_effects.csv")
    print("group score effects (validation):", {g: round(v, 3) for g, v in offsets.items()})


def select() -> None:
    rows = []
    for f in sorted(GRID.glob("*/train_history.csv")):
        h = pd.read_csv(f)
        best = h.loc[h.dev_hard_max_abs_gap.idxmin()]
        rows.append({"setting": f.parent.name, "best_epoch": int(best.epoch),
                     "dev_largest_gap": best.dev_hard_max_abs_gap,
                     **{c.replace("dev_hard_gap_", "dev_gap_"): best[c] for c in h.columns
                        if c.startswith("dev_hard_gap_")}})
    if not rows:
        raise SystemExit(f"no trained settings under {GRID}")
    table = pd.DataFrame(rows).sort_values("dev_largest_gap")
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "correction_settings_validation.csv", index=False)
    (GRID / "chosen.txt").write_text(table.iloc[0].setting + "\n")
    print(table.round(4).to_string(index=False))
    print("chosen:", table.iloc[0].setting)


def summary() -> None:
    have = {name: pd.read_csv(d / "per_pair.csv") for name, d in SOURCES.items() if (d / "per_pair.csv").exists()}
    if not have:
        raise SystemExit("no per-pair results yet")
    orig, chosen = "learned correction, original objective", "learned correction, setting chosen on validation"
    if orig in have and chosen in have and have[orig].new_outcome.equals(have[chosen].new_outcome):
        del have[chosen]
        print("The setting chosen on validation is the original one (same crops), reported once.")
    s = summarize_methods(have, "test")
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "summary_test.csv", index=False)
    view = s[s.value.isin(["argmax", "fixed"]) & s.measure.isin(COMPS + [GENDER, RACE, LARGEST])]
    wide = view.assign(points=100 * view.est).pivot_table(index=["method", "value"], columns="measure", values="points")
    print(wide[COMPS + [GENDER, RACE, LARGEST]].round(1).to_string())
    print(f"Saved {OUT / 'summary_test.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("bound", "select", "summary"))
    {"bound": bound, "select": select, "summary": summary}[ap.parse_args().action]()


if __name__ == "__main__":
    main()
