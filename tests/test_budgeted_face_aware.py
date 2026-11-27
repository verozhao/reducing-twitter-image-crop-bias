"""Unit tests for offline group-blind budgeted face-aware policy."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cropbias.geometry import BOTH, ONLY1, ONLY2
from scripts.budgeted_face_aware import _fairness, _record, apply_policy


def fake_records():
    return pd.DataFrame({
        "pair_id": ["t_001", "t_002", "t_003", "t_004"],
        "arg_outcome": [ONLY1, ONLY1, ONLY2, BOTH],
        "arg_quality": [.80, .80, .75, .65],
        "square_feasible": [True, True, False, True],
        "best_both_quality": [.79, .50, np.nan, .70],
        "arg_min_face_px": [0, 0, 0, 30],
        "best_both_min_face_px": [45, 50, np.nan, 40],
        "full_min_face_px": [35, 55, 45, 80],
    })


def test_quality_bound_and_argmax_both_not_replaced():
    rec = fake_records()
    out = apply_policy(rec, delta=.02, fallback_budget=0, min_face_px=32)
    assert out["mode"].tolist() == [1, 0, 0, 0]
    assert out["outcome"].tolist() == [BOTH, ONLY1, ONLY2, BOTH]
    assert out.loc[0, "square_quality"] == .79
    assert out.loc[1, "square_quality"] == .80
    assert out.loc[3, "square_quality"] == .65


def test_fallback_budget_is_global_bounded_and_ranked_by_face_size():
    rec = fake_records()
    # One of four composites may use full-image fallback. Among remaining
    # exclusive crops, record t_002 has wider readable full-image faces.
    out = apply_policy(rec, delta=.02, fallback_budget=.25, min_face_px=32)
    assert out["mode"].tolist() == [1, 2, 0, 0]
    assert np.isnan(out.loc[1, "square_quality"])
    assert out.loc[1, "min_displayed_face_px"] == 55
    assert (out["mode"] == 2).sum() <= int(.25 * len(out))


def test_unreadable_fallback_disallowed_even_when_budget_large():
    rec = fake_records()
    rec["full_min_face_px"] = 22.0
    out = apply_policy(rec, delta=0, fallback_budget=1, min_face_px=32)
    assert not (out["mode"] == 2).any()


def test_group_blind_selection_does_not_require_demographics():
    rec = fake_records()  # no race/gender columns
    out = apply_policy(rec, delta=.03, fallback_budget=.5, min_face_px=32)
    assert len(out) == len(rec)
    assert set(out["mode"]).issubset({0, 1, 2})


def test_gap_formula_signed_and_no_omitted_comparison():
    gaps = {"BF-BM": .10, "WF-BF": .02, "BF-WM": .11,
            "WF-BM": .10, "WM-BM": -.04, "WF-WM": .12}
    out = _fairness(gaps)
    assert np.isclose(out["gender_gap_pp"], 11)
    assert np.isclose(out["race_matched_gap_pp"], -1)
    assert np.isclose(out["max_six_abs_gap_pp"], 12)


def test_narrow_by_one_pixel_is_supported():
    # Validation contains W=254/255 and H=256 in a handful of real composites.
    xs = np.array([10., 30., 150., 240.], dtype=np.float32)
    sal = SimpleNamespace(xs=xs, S=np.array([[2., 0., 0., 0.],
                                             [0., 1., 0., 0.]], dtype=np.float32),
                          salient=(10, 100))
    lay = SimpleNamespace(W=255, H=256, f1x0=15., f1x1=55.,
                          f2x0=200., f2x1=245.)
    row = _record("val_demo_00000", lay, sal)
    assert row["square_feasible"] is False
    assert row["arg_outcome"] == BOTH
    assert row["arg_min_face_px"] > 0
