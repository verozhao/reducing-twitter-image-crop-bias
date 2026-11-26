"""Independent face-preserving square-preview policy.

Search *all pixel-aligned square windows*, not only saliency-grid centers. If a
square keeps both detected faces under the project's 90%-width criterion, keep
the one with the largest Twitter saliency mass. Otherwise letterbox the entire
composite into a square. This changes the display policy, not Twitter's model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import config as C
from .geometry import BOTH, inside_fraction, outcome, window
from .rules import View


@dataclass(frozen=True)
class Choice:
    fallback: bool
    feasible: bool
    x0: int | None
    x1: int | None
    face1_coverage: float
    face2_coverage: float
    saliency_retained: float
    deepgaze_retained: float
    scale: float
    padding_fraction: float
    outcome: int = BOTH


def choose(v: View) -> Choice:
    """Choose an inclusive-of-both square; otherwise use a full-image preview.

    All candidates have full image height (the composites must be at least as
    wide as they are tall). Tie breaks toward Twitter's original crop.
    """
    if v.W < v.H or v.crop_w != v.H:
        raise ValueError("face-preserving policy requires a wide composite and square crop")
    starts = np.arange(v.W - v.H + 1, dtype=np.int64)
    ends = starts + v.H
    c1 = inside_fraction(v.face1, starts, ends)
    c2 = inside_fraction(v.face2, starts, ends)
    ok = (c1 >= C.KEEP_FRACTION) & (c2 >= C.KEEP_FRACTION)
    if not np.any(ok):
        scale = v.H / v.W
        # Actual rendered height rounds to an integer, including the empty rows.
        rendered_height = max(1, int(round(v.H * scale)))
        return Choice(True, False, None, None, 1.0, 1.0, 1.0,
                      1.0 if v.dg_cols is not None else float("nan"),
                      scale, 1.0 - rendered_height / v.H)

    candidates = starts[ok]
    # Exact mass of saved saliency grid points within each pixel-aligned window.
    prefix = np.r_[0.0, np.cumsum(v.mass, dtype=np.float64)]
    left = np.searchsorted(v.xs, candidates, side="left")
    right = np.searchsorted(v.xs, candidates + v.H, side="left")
    masses = prefix[right] - prefix[left]
    maximum = masses.max()
    tied = candidates[np.isclose(masses, maximum, rtol=0.0, atol=1e-12)]
    arg_x0, _ = window(v.salient_x, v.H, v.W)
    x0 = int(tied[np.argmin(np.abs(tied - int(arg_x0)))])
    x1 = x0 + v.H
    dg = float(v.dg_cols[x0:x1].sum()) if v.dg_cols is not None else float("nan")
    assert outcome(v.face1, v.face2, x0, x1) == BOTH
    return Choice(False, True, x0, x1,
                  float(inside_fraction(v.face1, x0, x1)),
                  float(inside_fraction(v.face2, x0, x1)),
                  float(prefix[np.searchsorted(v.xs, x1, side="left")] -
                        prefix[np.searchsorted(v.xs, x0, side="left")]),
                  dg, 1.0, 0.0)


def render(src: Path, choice: Choice, dest: Path, background=(32, 32, 32)) -> None:
    """Save the actual square preview; fallback fits the *whole* image with padding."""
    with Image.open(src) as f:
        img = f.convert("RGB")
    if choice.fallback:
        side = img.height
        fitted_height = max(1, int(round(side * side / img.width)))
        fitted = img.resize((side, fitted_height), Image.Resampling.LANCZOS)
        preview = Image.new("RGB", (side, side), background)
        preview.paste(fitted, (0, (side - fitted_height) // 2))
    else:
        if choice.x0 is None or choice.x1 is None or choice.x1 - choice.x0 != img.height:
            raise ValueError("square choice does not match image height")
        preview = img.crop((choice.x0, 0, choice.x1, img.height))
    dest.parent.mkdir(parents=True, exist_ok=True)
    preview.save(dest, "JPEG", quality=C.JPEG_QUALITY)
