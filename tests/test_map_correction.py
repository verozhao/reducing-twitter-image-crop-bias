"""Small checks for the new experiment; no private photos or Twitter binary needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cropbias import config as C
from cropbias.map_correction import (SIZE, _batch_losses, correction_model, crop_stats,
                                     person_partition, transform)
from cropbias.twitter_saliency import SalOut


def test_matched_preprocessing_preserves_geometry_and_matches_luminance():
    rng = np.random.default_rng(44)
    a = rng.integers(18, 80, size=(40, 20, 3), dtype=np.uint8)
    b = rng.integers(125, 215, size=(40, 20, 3), dtype=np.uint8)
    image = Image.fromarray(np.concatenate((a, b), axis=1))
    assert transform(image, 20, "gray").size == image.size
    out = np.asarray(transform(image, 20, "matched_luminance"), dtype=float)
    y1 = np.sum(out[:, :20] * [0.2126, 0.7152, 0.0722], axis=2)
    y2 = np.sum(out[:, 20:] * [0.2126, 0.7152, 0.0722], axis=2)
    assert abs(y1.mean() - y2.mean()) < 3


def test_correction_changes_map_before_same_argmax_crop():
    scores = np.array([[1, 0, 9, 0], [1, 0, 1, 0]], dtype=np.float32)
    sal = SalOut((68, 1), np.zeros((1, 4), dtype=np.int32),
                 np.array([20, 45, 68, 95], np.float32),
                 np.array([1, 2], np.float32), scores)
    lay = pd.Series({"W": 120, "H": 40, "f1x0": 10, "f1x1": 30,
                     "f2x0": 70, "f2x1": 90})
    assert crop_stats(lay, sal)["outcome"] == 2
    corrected = np.array([[9, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    assert crop_stats(lay, sal, corrected)["outcome"] == 1


def test_partitions_people_not_pairs():
    people, rows = [], []
    for group in C.GROUPS:
        people += [{"qid": f"{group}-{i}", "group": group, "split": "val"} for i in range(12)]
    for a, b in C.COMPARISONS:
        for i in range(144):
            rows.append({"comparison": f"{a}-{b}", "q1": f"{a}-{i//12}",
                         "q2": f"{b}-{i%12}"})
    fit, dev = person_partition(pd.DataFrame(rows), pd.DataFrame(people))
    assert set(fit.q1).isdisjoint(dev.q1)
    assert set(fit.q2).isdisjoint(dev.q2)
    assert len(set(fit.comparison)) == len(set(dev.comparison)) == 6


def test_correction_has_gradient_and_zero_init():
    torch = pytest.importorskip("torch")
    model = correction_model()
    X = torch.randn(3, 4, SIZE[1], SIZE[0])
    S = torch.randn(3, SIZE[1], SIZE[0])
    M = torch.sign(torch.randn(3, SIZE[0]))
    assert torch.count_nonzero(model(X, 1.0)) == 0
    signed, fidelity, smooth = _batch_losses(model, X, S, M, 1.0, 0.2)
    loss = signed.square().mean() + .1*fidelity + .03*smooth
    loss.backward()
    assert torch.isfinite(loss)
    assert model.net[-2].weight.grad is not None
    assert torch.isfinite(model.net[-2].weight.grad).all()
