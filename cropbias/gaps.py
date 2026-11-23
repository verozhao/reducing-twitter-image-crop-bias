"""Score gap between the top two crop candidates, used to set the tie threshold.

The tie threshold eps is chosen on validation composites so that a fixed share of them
counts as tied. This needs no demographic labels and no face boxes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .geometry import crop_width
from .rules import find_peaks
from .twitter_saliency import load_store


def top_two_gap(S: np.ndarray, xs: np.ndarray, crop_w: int) -> float:
    m = S.max(axis=0).astype(np.float64)
    pk = find_peaks(m, xs, crop_w / 2.0)
    return float(m[pk[0]] - m[pk[1]]) if len(pk) > 1 else np.inf


def split_gaps(split: str, layout: pd.DataFrame) -> dict[str, np.ndarray]:
    store = load_store(split, "orig")
    out = {a: [] for a in C.ASPECTS}
    for pid, lay in layout.iterrows():
        s = store[pid]
        for a, ar in C.ASPECTS.items():
            out[a].append(top_two_gap(s.S, s.xs, crop_width(int(lay.H), int(lay.W), ar)))
    return {a: np.array(v) for a, v in out.items()}
