"""Pure unit checks for the square-stage constrained controls."""
import numpy as np
import pandas as pd
import pytest

from scripts.square_stage_random_control import (
    _gap_values, _random_candidates, method_summary, random_draw_summary,
)


def test_uniform_candidates_exclude_original_and_preserve_edit_budget():
    x = np.arange(6, dtype=float)
    sal = np.zeros((1, 6))
    pools = _random_candidates(6, 3, (2.2, 2.9), (3.5, 4.2), x, sal,
                               original_x0=0, original_quality=.5,
                               faceaware_quality=.5, delta=0, tolerance=0)
    assert pools["uniform_feasible"]["n_candidates"] == 3
    assert pools["uniform_quality_matched"]["n_candidates"] == 3
    np.testing.assert_allclose(pools["uniform_feasible"]["expected_quality"], .5)
    assert pools["uniform_feasible"]["prob_both"] == pytest.approx(1/3)


def test_quality_match_enforces_per_image_tolerance():
    xs = np.arange(7, dtype=float)
    s = np.array([[0., 0., 0., 0.5, 0., 0., 0.]])
    mass = np.exp(s - s.max()); mass /= mass.sum()
    q = np.array([mass[0, j:j+3].sum() for j in range(5)])
    # The alternative both-face square x0=2 has known quality q[2].
    pools = _random_candidates(7, 3, (2.1, 2.7), (3.4, 4.0), xs, s,
                               original_x0=0, original_quality=float(q[0]),
                               faceaware_quality=float(q[2]), delta=.5, tolerance=1e-10)
    assert pools["uniform_quality_matched"]["n_candidates"] > 0
    assert np.all(np.abs(pools["uniform_quality_matched"]["quality"] - q[2]) < 1e-9)
    assert pools["uniform_feasible"]["n_candidates"] >= pools["uniform_quality_matched"]["n_candidates"]


def test_rejects_stale_cache_quality():
    with pytest.raises(ValueError, match="does not match"):
        _random_candidates(6, 3, (2, 3), (3, 4), np.arange(6, dtype=float),
                           np.zeros((1, 6)), 0, .2, .5, .1, .005)


def test_gap_convention_unchanged():
    from scripts.square_stage_random_control import COMPS
    contrib = np.array([.1, .2, -.1, 0, .3, .4])
    g, r, m, means = _gap_values(contrib, np.arange(6))
    assert g == pytest.approx(25.0)
    assert r == pytest.approx(25.0)
    assert m == pytest.approx(40.0)


def test_no_private_pair_ids_in_summary():
    # Tiny synthetic example needs all six comparisons.
    from scripts.square_stage_random_control import COMPS
    pairs = pd.DataFrame({'comparison': COMPS})
    rec = pd.DataFrame({'arg_quality':[.8]*6})
    template = {'gap':np.zeros(6), 'quality':np.full(6,.8),
                'width':np.full(6,30.),'both':np.zeros(6),'p_width32':np.zeros(6)}
    methods = {'argmax':template,'face_aware_square':template}
    for x in ('uniform_feasible','uniform_quality_matched'):
        methods[x] = {**template,'n_candidates':[2,3], 'draw_change':np.zeros((2,6), dtype=np.float32)}
    result = method_summary(methods, np.array([0,1]),pairs,rec,.005)
    assert result.shape[0] == 4
    assert 'pair_id' not in result.columns


def test_per_comparison_reports_all_six_pairs():
    from scripts.square_stage_random_control import COMPS, per_comparison_summary
    pairs = pd.DataFrame({'comparison': COMPS})
    state = {'gap':np.zeros(6), 'quality':np.ones(6), 'both':np.ones(6)}
    result = per_comparison_summary({'method':state},pairs)
    assert result.shape == (6, 6)
    assert set(result.comparison) == set(COMPS)
    assert (result['signed_gap_pp'] == 0).all()
