"""Side-by-side composites and the small changes used for stability fallback.

Both photos are scaled to the same height and joined left to right, as in the
fixed-height setting of Twitter's audit. All x coordinates are composite pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

from . import config as C

COMPOSITE_DIR = C.WORK / "composites"


@dataclass
class Layout:
    W: int
    H: int
    w_left: int
    face1: tuple[float, float]  # x interval of the group-1 person's face box
    face2: tuple[float, float]  # x interval of the group-2 person's face box
    face1_y: tuple[float, float]
    face2_y: tuple[float, float]


def scaled(img: Image.Image, height: int) -> Image.Image:
    w = max(1, int(round(img.width * height / img.height)))
    return img.resize((w, height), Image.LANCZOS)


def compose(left: Image.Image, right: Image.Image, height: int = C.COMPOSITE_HEIGHT) -> tuple[Image.Image, int]:
    a, b = scaled(left, height), scaled(right, height)
    out = Image.new("RGB", (a.width + b.width, height))
    out.paste(a, (0, 0))
    out.paste(b, (a.width, 0))
    return out, a.width


def build(pair: pd.Series, faces: pd.DataFrame, height: int = C.COMPOSITE_HEIGHT) -> tuple[Image.Image, Layout]:
    """faces is indexed by qid with columns x1, y1, x2, y2, img_w, img_h."""
    q_left, q_right = (pair.q1, pair.q2) if pair.g1_left else (pair.q2, pair.q1)
    left = Image.open(C.IMAGES / f"{q_left}.jpg").convert("RGB")
    right = Image.open(C.IMAGES / f"{q_right}.jpg").convert("RGB")
    img, w_left = compose(left, right, height)
    boxes = {}
    for q, offset in ((q_left, 0), (q_right, w_left)):
        f = faces.loc[q]
        s = height / f.img_h
        boxes[q] = ((f.x1 * s + offset, f.x2 * s + offset), (f.y1 * s, f.y2 * s))
    lay = Layout(W=img.width, H=height, w_left=w_left,
                 face1=boxes[pair.q1][0], face2=boxes[pair.q2][0],
                 face1_y=boxes[pair.q1][1], face2_y=boxes[pair.q2][1])
    return img, lay


# ----------------------------------------------------------------------------
# Perturbations. Each returns the changed image and a function that maps an x
# interval of the original composite into the changed image's frame.
# ----------------------------------------------------------------------------
def _shift(img: Image.Image, dx: int) -> Image.Image:
    a = np.asarray(img)
    out = np.empty_like(a)
    if dx > 0:
        out[:, dx:] = a[:, :-dx]
        out[:, :dx] = a[:, :1]
    elif dx < 0:
        out[:, :dx] = a[:, -dx:]
        out[:, dx:] = a[:, -1:]
    else:
        out[:] = a
    return Image.fromarray(out)


def shift_px(W: int) -> int:
    return max(1, int(round(C.PERTURB_SHIFT_FRAC * W)))


def perturb(img: Image.Image, variant: str) -> Image.Image:
    W = img.width
    if variant == "orig":
        return img
    if variant == "mirror":
        return ImageOps.mirror(img)
    if variant == "bright":
        return ImageEnhance.Brightness(img).enhance(1 + C.PERTURB_BRIGHTNESS)
    if variant == "dark":
        return ImageEnhance.Brightness(img).enhance(1 - C.PERTURB_BRIGHTNESS)
    if variant == "shift_r":
        return _shift(img, shift_px(W))
    if variant == "shift_l":
        return _shift(img, -shift_px(W))
    if variant == "mirror_shift":
        return _shift(ImageOps.mirror(img), shift_px(W))
    raise ValueError(variant)


def map_interval(iv: tuple[float, float], variant: str, W: int) -> tuple[float, float]:
    """Where a face's x interval ends up in the changed image (clipped to the frame)."""
    x0, x1 = iv
    dx = shift_px(W)
    if variant in ("orig", "bright", "dark"):
        a, b = x0, x1
    elif variant == "mirror":
        a, b = W - x1, W - x0
    elif variant == "shift_r":
        a, b = x0 + dx, x1 + dx
    elif variant == "shift_l":
        a, b = x0 - dx, x1 - dx
    elif variant == "mirror_shift":
        a, b = W - x1 + dx, W - x0 + dx
    else:
        raise ValueError(variant)
    return max(0.0, a), min(float(W), b)


def composite_path(pair_id: str, split: str) -> Path:
    return COMPOSITE_DIR / split / f"{pair_id}.jpg"
