import numpy as np
import pandas as pd

from cropbias import config as C
from cropbias.score_fixes import (COMPS, GENDER, LARGEST, RACE, column_shift, method_rows,
                                  offsets_from_margins)


def test_fairness_term_mean_square_and_maximin():
    import torch
    from cropbias.map_correction import fairness_term
    gaps = [torch.tensor(0.1), torch.tensor(-0.3), torch.tensor(0.2)]
    assert np.isclose(float(fairness_term(gaps, "mean_square")), (0.01 + 0.09 + 0.04) / 3)
    assert np.isclose(float(fairness_term(gaps, "maximin")), 0.09)


def test_group_effects_are_recovered_from_margins():
    rng = np.random.default_rng(0)
    truth = {"BF": 0.3, "BM": -0.2, "WF": 0.25, "WM": -0.35}       # centered: sums to zero
    g1 = rng.choice(C.GROUPS, 4000)
    g2 = np.array([rng.choice([g for g in C.GROUPS if g != a]) for a in g1])
    margin = np.array([truth[a] - truth[b] for a, b in zip(g1, g2)]) + rng.normal(0, 0.05, len(g1))
    est = offsets_from_margins(g1, g2, margin)
    for g in C.GROUPS:
        assert abs(est[g] - truth[g]) < 0.01


def test_each_half_of_the_map_loses_its_own_groups_effect():
    shift = column_shift(np.array([10.0, 60.0, 140.0, 190.0]), 100.0, "WF", "BM", {"WF": 0.5, "BM": -0.2})
    assert shift.tolist() == [0.5, 0.5, -0.2, -0.2]


def test_pooled_measures_average_the_right_comparisons():
    base = {"BF-BM": 0.10, "WF-BF": 0.02, "BF-WM": 0.12, "WF-BM": 0.14, "WM-BM": -0.04, "WF-WM": 0.11}
    new = {"BF-BM": 0.00, "WF-BF": 0.08, "BF-WM": 0.02, "WF-BM": 0.10, "WM-BM": 0.04, "WF-WM": 0.06}
    comp = np.repeat(COMPS, 4)
    df = pd.DataFrame({"baseline_gap": [base[c] for c in comp], "new_gap": [new[c] for c in comp],
                       "baseline_outcome": 1, "new_outcome": [0, 1, 1, 1] * len(COMPS)})
    rows = pd.DataFrame(method_rows("fix", df, comp, np.ones((5, len(comp)))))
    get = lambda m, v: rows[(rows.measure == m) & (rows.value == v)].est.iloc[0]
    assert np.isclose(get(GENDER, "argmax"), (0.10 + 0.11 + 0.12 + 0.14) / 4)
    assert np.isclose(get(RACE, "fixed"), (0.08 + 0.04) / 2)
    assert np.isclose(get(RACE, "change"), (0.08 + 0.04) / 2 - (0.02 - 0.04) / 2)
    assert np.isclose(get(LARGEST, "argmax"), 0.14) and np.isclose(get(LARGEST, "fixed"), 0.10)
    assert np.isclose(get("share of crops changed", "fixed"), 0.25)
    assert np.isclose(get("share of crops with no face", "fixed"), 0.25)
