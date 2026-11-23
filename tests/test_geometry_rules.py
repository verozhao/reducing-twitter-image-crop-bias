import numpy as np
import pytest

from cropbias.geometry import BOTH, NEITHER, ONLY1, ONLY2, crop_width, outcome, window
from cropbias.rules import (View, find_peaks, r_argmax, r_random_fallback, r_sampling, r_stability,
                            r_tie, r_topk, softmax)
from cropbias.twitter_saliency import parse


def twitter_generate_crop_x(img_h, img_w, x, target_ratio):
    """Copy of Twitter's crop_api.generate_crop / fit_window (x part only)."""
    image_ratio = img_h / img_w
    if target_ratio < image_ratio:
        return 0, img_w
    width = np.round(img_h / target_ratio)
    fr = x - width // 2
    to = fr + width
    if fr < 0:
        fr, to = 0, width
    elif to > img_w:
        to = img_w
        fr = to - width
    return int(fr), int(to)


@pytest.mark.parametrize("aspect", [1.0, 4 / 3])
def test_window_matches_twitter(aspect):
    rng = np.random.default_rng(0)
    for _ in range(500):
        W = int(rng.integers(300, 700))
        H = 256
        x = int(rng.integers(0, W))
        cw = crop_width(H, W, aspect)
        x0, x1 = window(x, cw, W)
        assert (int(x0), int(x1)) == twitter_generate_crop_x(H, W, x, aspect)


def test_crop_width_narrow_image_keeps_full_width():
    assert crop_width(256, 200, 1.0) == 200
    assert crop_width(256, 400, 1.0) == 256
    assert crop_width(256, 400, 4 / 3) == 192


def test_outcome_codes():
    f1, f2 = (60.0, 140.0), (260.0, 340.0)
    assert outcome(f1, f2, 0, 256) == ONLY1
    assert outcome(f1, f2, 144, 400) == ONLY2
    assert outcome(f1, f2, 0, 400) == BOTH
    assert outcome(f1, f2, 120, 280) == NEITHER
    # 90% rule: 72 of 80 pixels inside is kept, 71 is not
    assert outcome(f1, f2, 68, 324) & 1
    assert not (outcome(f1, f2, 69, 325) & 1)


def make_view(s1=1.0, s2=0.9, W=400, H=256, aspect=1.0, salient=None):
    xs = np.arange(10, W, 20, dtype=np.float32)
    ys = np.arange(10, H, 20, dtype=np.float32)
    S = np.full((len(ys), len(xs)), -1.0, np.float32)
    S[6, np.argmin(np.abs(xs - 100))] = s1
    S[6, np.argmin(np.abs(xs - 300))] = s2
    if salient is None:
        salient = int(np.floor(xs[np.unravel_index(np.argmax(S), S.shape)[1]]))
    return View(W, H, crop_width(H, W, aspect), (60.0, 140.0), (260.0, 340.0), xs, S, salient)


def test_argmax_and_ties():
    v = make_view()
    assert r_argmax(v).p.tolist() == [0, 1, 0, 0]
    assert r_tie(v, 0.05).p.tolist() == [0, 1, 0, 0]
    assert np.allclose(r_tie(v, 0.2).p, [0, 0.5, 0.5, 0])


def test_peaks_are_separated_and_sorted():
    v = make_view()
    pk = v.peaks
    assert np.floor(v.xs[pk[0]]) == 90
    assert all(abs(v.xs[a] - v.xs[b]) >= v.crop_w / 2 for i, a in enumerate(pk) for b in pk[i + 1:])
    assert np.all(np.diff(v.m[pk]) <= 0)


def test_sampling_exact_matches_monte_carlo():
    v = make_view(1.0, 0.8)
    T = 0.3
    d = r_sampling(v, T)
    assert np.isclose(d.p.sum(), 1)
    P = softmax(v.S.ravel() / T)
    rng = np.random.default_rng(1)
    draws = rng.choice(len(P), size=200_000, p=P)
    cols = draws % v.S.shape[1]
    mc = np.bincount(v.col_out[cols], minlength=4) / len(cols)
    assert np.allclose(d.p, mc, atol=0.01)
    assert np.allclose(r_sampling(v, 0.001).p, [0, 1, 0, 0])


def test_random_fallback_and_stability():
    v = make_view()
    d = r_random_fallback(v, 0.3)
    assert np.allclose(d.p, [0, 0.7, 0, 0.3]) and d.fb == 0.3
    same, flipped = make_view(), make_view(0.9, 1.0)
    assert r_stability(v, [same, flipped, same, same], 0).fb == 1.0
    assert r_stability(v, [same, flipped, same, same], 1).fb == 0.0


def test_topk():
    v = make_view()
    d1 = r_topk(v, 1, "ordered")
    assert d1.p.tolist() == [0, 1, 0, 0]
    d2 = r_topk(v, 2, "ordered")
    assert d2.p.tolist() == [0, 0, 0, 1] and d2.p_first.tolist() == [0, 1, 0, 0]
    d2s = r_topk(v, 2, "shuffled")
    assert np.allclose(d2s.p_first, [0, 0.5, 0.5, 0])
    d2r = r_topk(v, 2, "sampled", T=0.1, seed_key="x", draws=4000)
    assert d2r.p[BOTH] > 0.5 and np.isclose(d2r.p.sum(), 1)


def test_parse_program_output():
    out = "129 336\n0 0 851 851\n0 0 638 851\n" + "\n".join(
        f"{x + 0.5:.2f} {y + 0.5:.2f} {0.01 * x + y:.4f}" for y in range(3) for x in range(4))
    s = parse(out, 2)
    assert s.salient == (129, 336) and s.crops.shape == (2, 4)
    assert s.S.shape == (3, 4) and np.isclose(s.S[2, 3], 2.03)
