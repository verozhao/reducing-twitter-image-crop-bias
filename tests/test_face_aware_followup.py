"""Regression tests for the controlled face-aware follow-up."""
import numpy as np
import pandas as pd
import pytest

from cropbias.geometry import BOTH, ONLY1, ONLY2
from scripts.budgeted_face_aware import apply_policy
from scripts.face_aware_followup import (
    argmax_policy, calibrate_threshold, fallback_eligible,
    online_policy, random_policy,
)


def fake_records():
    # 2 faces in square for first record; three available fallback images.
    return pd.DataFrame({
        "pair_id": ["a", "b", "c", "d", "e", "f"],
        "arg_outcome": [ONLY1, ONLY1, ONLY2, ONLY1, ONLY2, BOTH],
        "arg_quality": [.8, .8, .8, .8, .8, .8],
        "square_feasible": [True, False, False, False, False, True],
        "best_both_quality": [.8, np.nan, np.nan, np.nan, np.nan, .8],
        "arg_min_face_px": [0, 0, 0, 0, 0, 40],
        "best_both_min_face_px": [42, np.nan, np.nan, np.nan, np.nan, 40],
        "full_min_face_px": [50, 55, 45, 35, 20, 66],
    })


def test_calibration_uses_validation_threshold_and_eligibility():
    rec = fake_records()
    tau, desc = calibrate_threshold(rec, delta=.0, budget=1/3, min_face_px=32)
    assert desc["target_fallback_count_val"] == 2
    assert 35 < tau < 45
    assert desc["calibrated_fallback_count_val"] == 2
    assert tau == 40


def test_online_is_per_image_and_not_exact_quota():
    val = fake_records()
    tau, _ = calibrate_threshold(val, 0, 1/3, 32)
    test = fake_records()
    test.loc[3, "full_min_face_px"] = 50
    out = online_policy(test, 0, tau, 32)
    # The test distribution changed: now three images meet the same frozen threshold.
    assert (out["mode"] == 2).sum() == 3
    assert (out["mode"] == 1).sum() == 1
    assert (out.loc[out["mode"] == 2, "outcome"] == BOTH).all()


def test_random_control_is_budget_and_eligibility_matched():
    rec = fake_records()
    sq = apply_policy(rec, 0, 0, 32)
    elig = fallback_eligible(rec, sq, 32)
    rand = random_policy(rec, sq, 2, 32, seed=123)
    assert int((rand["mode"] == 2).sum()) == 2
    assert not np.any((rand["mode"].to_numpy() == 2) & ~elig)
    np.testing.assert_array_equal(rand["mode"], random_policy(rec, sq, 2, 32, seed=123)["mode"])
    np.testing.assert_array_equal(rand.loc[0:0, "mode"], np.array([1]))


def test_baseline_does_not_inherit_square_changes():
    rec = fake_records()
    sq = apply_policy(rec, 0, 0, 32)
    orig = argmax_policy(rec, sq)
    assert (orig["mode"] == 0).all()
    np.testing.assert_array_equal(orig["outcome"], rec["arg_outcome"])


def test_random_excess_raises():
    rec = fake_records()
    sq = apply_policy(rec, 0, 0, 32)
    with pytest.raises(ValueError, match="eligibility pool"):
        random_policy(rec, sq, 9, 32, seed=1)
