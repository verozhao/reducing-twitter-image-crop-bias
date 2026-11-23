"""Crop decision rules. Every rule reads the same saliency grid and only changes how a
crop center is picked from it.

Rules return the full probability of each outcome (neither / only group-1 person /
only group-2 person / both), computed exactly where possible, so random rules need no
seeds and add no sampling noise. A fallback (showing the full image) counts as "both".
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .geometry import BOTH, NEITHER, crop_width, cut, outcome, window


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class View:
    """One composite (or one changed copy) seen at one crop aspect ratio."""
    W: int
    H: int
    crop_w: int
    face1: tuple[float, float]
    face2: tuple[float, float]
    xs: np.ndarray
    S: np.ndarray
    salient_x: int
    dg_cols: np.ndarray | None = None   # DeepGaze density summed over rows, one value per pixel column
    # derived fields
    centers: np.ndarray = field(init=False)
    col_out: np.ndarray = field(init=False)
    col_cut: np.ndarray = field(init=False)
    col_tw: np.ndarray = field(init=False)
    col_dg: np.ndarray | None = field(init=False)
    m: np.ndarray = field(init=False)
    peaks: np.ndarray = field(init=False)
    arg: dict = field(init=False)

    def __post_init__(self):
        self.centers = np.floor(self.xs).astype(np.int64)
        x0, x1 = window(self.centers, self.crop_w, self.W)
        self.col_out = outcome(self.face1, self.face2, x0, x1)
        self.col_cut = cut(self.face1, self.face2, x0, x1).astype(np.float64)
        # Share of Twitter saliency per grid column (scores treated as logits, as in
        # Twitter's own sampling code).
        self.mass = softmax(self.S.astype(np.float64).ravel()).reshape(self.S.shape).sum(axis=0)
        inside = (self.xs[None, :] >= x0[:, None]) & (self.xs[None, :] < x1[:, None])
        self.col_tw = inside @ self.mass
        if self.dg_cols is not None:
            cs = np.r_[0.0, np.cumsum(self.dg_cols)]
            self.col_dg = cs[x1] - cs[x0]
        else:
            self.col_dg = None
        self.m = self.S.max(axis=0).astype(np.float64)
        self.peaks = find_peaks(self.m, self.xs, self.crop_w / 2.0)
        self.arg = self.at(self.salient_x)

    def at(self, center_x: float) -> dict:
        """Outcome and quality of the crop centered at one x."""
        x0, x1 = window(center_x, self.crop_w, self.W)
        x0, x1 = int(x0), int(x1)
        inside = (self.xs >= x0) & (self.xs < x1)
        dg = None
        if self.dg_cols is not None:
            dg = float(self.dg_cols[x0:x1].sum())
        return {"out": int(outcome(self.face1, self.face2, x0, x1)),
                "cut": float(cut(self.face1, self.face2, x0, x1)),
                "tw": float(self.mass[inside].sum()), "dg": dg}


def find_peaks(m: np.ndarray, xs: np.ndarray, min_dist: float) -> np.ndarray:
    """Local maxima of the column profile, greedily kept if at least min_dist apart.
    Sorted from highest to lowest score. The first peak is the argmax column."""
    n = len(m)
    left = np.r_[-np.inf, m[:-1]]
    right = np.r_[m[1:], -np.inf]
    cand = np.where((m >= left) & (m >= right))[0]
    order = cand[np.argsort(-m[cand], kind="stable")]
    top = int(np.argmax(m))
    if top not in order:  # flat top edge case
        order = np.r_[top, order]
    keep: list[int] = []
    for k in order:
        if all(abs(xs[k] - xs[j]) >= min_dist for j in keep):
            keep.append(int(k))
    return np.array(keep, dtype=np.int64)


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class Dist:
    p: np.ndarray            # (4,) outcome probabilities (fallback counted as both)
    fb: float = 0.0          # probability of showing the full image
    tw: float = np.nan       # expected share of Twitter saliency inside the preview
    dg: float = np.nan       # expected share of DeepGaze IIE density inside the preview
    cut: float = np.nan      # probability that some face is cut by the crop edge
    p_first: np.ndarray | None = None  # top-k rules: outcome of the first suggestion


def onehot(o: int) -> np.ndarray:
    v = np.zeros(4)
    v[o] = 1.0
    return v


def _mix_columns(v: View, q: np.ndarray) -> Dist:
    """A rule that picks grid column k with probability q[k]."""
    p = np.bincount(v.col_out, weights=q, minlength=4)
    dg = float(q @ v.col_dg) if v.col_dg is not None else np.nan
    return Dist(p=p, tw=float(q @ v.col_tw), dg=dg, cut=float(q @ v.col_cut))


def _from_point(a: dict) -> Dist:
    return Dist(p=onehot(a["out"]), tw=a["tw"], dg=a["dg"] if a["dg"] is not None else np.nan, cut=a["cut"])


def r_argmax(v: View) -> Dist:
    return _from_point(v.arg)


def r_sampling(v: View, T: float) -> Dist:
    """Twitter's suggested alternative: draw the focal point with probability
    proportional to exp(score / T). Twitter's own sampling code used T = 1."""
    P = softmax(v.S.astype(np.float64).ravel() / T).reshape(v.S.shape)
    return _mix_columns(v, P.sum(axis=0))


def r_averaging(v: View, n: int) -> Dist:
    """Twitter's suggested alternative: center on the average x of the top-n points."""
    flat = v.S.ravel()
    idx = np.argsort(-flat, kind="stable")[:n]
    cols = idx % v.S.shape[1]
    return _from_point(v.at(float(v.xs[cols].mean())))


def r_tie(v: View, eps: float) -> Dist:
    """Ours: peaks within eps of the top peak count as tied; pick one uniformly."""
    pk = v.peaks
    tied = pk[v.m[pk] >= v.m[pk[0]] - eps]
    if len(tied) <= 1:
        return r_argmax(v)
    q = np.zeros(len(v.m))
    q[tied] = 1.0 / len(tied)
    return _mix_columns(v, q)


def full_image() -> Dist:
    return Dist(p=onehot(BOTH), fb=1.0, tw=1.0, dg=1.0, cut=0.0)


def r_stability(v: View, copies: list[View], max_changes: int) -> Dist:
    """Ours: run argmax on the photo and on changed copies. If the kept person changes in
    more than max_changes copies, show the full image."""
    o = v.arg["out"]
    changes = sum(int(c.arg["out"] != o) for c in copies)
    return full_image() if changes > max_changes else r_argmax(v)


def r_random_fallback(v: View, q: float) -> Dist:
    """Control: show the full image for a random share q of photos."""
    a = r_argmax(v)
    f = full_image()
    return Dist(p=(1 - q) * a.p + q * f.p, fb=q, tw=(1 - q) * a.tw + q * f.tw,
                dg=(1 - q) * a.dg + q * f.dg, cut=(1 - q) * a.cut + q * f.cut)


def _coverage_code(outs: np.ndarray) -> int:
    k1 = np.any((outs == 1) | (outs == 3))
    k2 = np.any((outs == 2) | (outs == 3))
    return int(k1) + 2 * int(k2)


def r_topk(v: View, k: int, mode: str = "ordered", T: float | None = None,
           seed_key: str = "", draws: int = 256) -> Dist:
    """Instructor's suggestion: offer k crop suggestions (distinct peaks).
    p is the coverage outcome (who appears in at least one suggestion),
    p_first the outcome of the first suggestion shown.
    ordered  - the k highest peaks, best first
    shuffled - the same k peaks in random order
    sampled  - k peaks drawn without replacement with weights exp(score / T)"""
    pk = v.peaks
    outs = v.col_out[pk]
    if mode in ("ordered", "shuffled"):
        sel = outs[:k]
        cov = onehot(_coverage_code(sel))
        if mode == "ordered":
            first = onehot(int(sel[0]))
        else:
            first = np.bincount(sel, minlength=4) / len(sel)
        q_first = np.zeros(len(v.m))
        if mode == "ordered":
            q_first[pk[0]] = 1.0
        else:
            q_first[pk[:k]] = 1.0 / len(pk[:k])
    elif mode == "sampled":
        rng = np.random.default_rng(zlib.crc32(f"topk:{seed_key}:{k}:{T}".encode()))
        g = rng.gumbel(size=(draws, len(pk))) + v.m[pk][None, :] / T
        order = np.argsort(-g, axis=1)[:, :k]
        picked = outs[order]
        k1 = np.any((picked == 1) | (picked == 3), axis=1)
        k2 = np.any((picked == 2) | (picked == 3), axis=1)
        cov = np.bincount(k1.astype(int) + 2 * k2.astype(int), minlength=4) / draws
        first = np.bincount(picked[:, 0], minlength=4) / draws
        q_first = np.bincount(pk[order[:, 0]], minlength=len(v.m)) / draws
    else:
        raise ValueError(mode)
    d = _mix_columns(v, q_first)  # quality of the first suggestion
    return Dist(p=cov, tw=d.tw, dg=d.dg, cut=d.cut, p_first=first)
