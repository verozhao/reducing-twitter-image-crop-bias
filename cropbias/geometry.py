"""Crop windows and who a crop keeps.

Twitter's crop logic (crop_api.py, generate_crop and fit_window): the crop is the
largest box with the target aspect ratio (height / width) that fits the image,
centered on the chosen point and pushed back inside the image at the edges. For our
wide composites the crop always has full height, so only its x range matters.
"""
from __future__ import annotations

import numpy as np

from . import config as C

NEITHER, ONLY1, ONLY2, BOTH = 0, 1, 2, 3
OUTCOMES = ["neither", "only_g1", "only_g2", "both"]


def crop_width(H: int, W: int, aspect: float) -> int:
    """Width of the crop window. Returns W when the crop keeps the full width."""
    if aspect < H / W:        # Twitter "squeezes vertically": full width, shorter height
        return W
    return int(min(W, np.round(H / aspect)))


def window(center_x, crop_w: int, W: int):
    """Twitter's fit_window on the x axis. Works on scalars and numpy arrays."""
    c = np.floor(np.asarray(center_x, dtype=np.float64)).astype(np.int64)
    fr = c - crop_w // 2
    fr = np.clip(fr, 0, W - crop_w)
    return fr, fr + crop_w


def inside_fraction(face: tuple[float, float], x0, x1):
    f0, f1 = face
    width = max(f1 - f0, 1e-9)
    ov = np.clip(np.minimum(x1, f1) - np.maximum(x0, f0), 0, None)
    return ov / width


def outcome(face1, face2, x0, x1):
    """0 neither, 1 only group-1 person, 2 only group-2 person, 3 both."""
    k1 = inside_fraction(face1, x0, x1) >= C.KEEP_FRACTION
    k2 = inside_fraction(face2, x0, x1) >= C.KEEP_FRACTION
    return (k1.astype(np.int64) * 1 + k2.astype(np.int64) * 2) if isinstance(k1, np.ndarray) \
        else int(k1) * 1 + int(k2) * 2


def cut(face1, face2, x0, x1):
    """True when some face is partly inside the crop (more than 10% but less than 90%)."""
    out = False
    for f in (face1, face2):
        r = inside_fraction(f, x0, x1)
        out = out | ((r > C.CUT_FRACTION) & (r < C.KEEP_FRACTION))
    return out
