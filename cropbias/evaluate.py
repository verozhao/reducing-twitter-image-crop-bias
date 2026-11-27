"""Apply every rule to every composite and keep per-composite results.

Output per split (work/eval/<split>.npz): an array [composite, config, metric] for each
crop aspect, plus a table of per-composite facts used by the near-tie analysis.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as C
from .composites import map_interval
from .geometry import crop_width, window
from .rules import (View, r_argmax, r_averaging, r_random_fallback, r_sampling, r_stability,
                    r_tie, r_topk)
from .twitter_saliency import load_store

EVAL_DIR = C.WORK / "eval"
METRICS = ["p_neither", "p_only1", "p_only2", "p_both", "fb", "tw", "dg", "cut",
           "pred_same", "pred_copy", "change",
           "f_neither", "f_only1", "f_only2", "f_both"]
TIE_RATES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]


@dataclass(frozen=True)
class Config:
    name: str
    family: str
    param: object = None


def build_configs(eps_grid: dict[float, float]) -> list[Config]:
    cfg = [Config("argmax", "argmax")]
    cfg += [Config(f"sampling_T{T:g}", "sampling", T) for T in (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)]
    cfg += [Config(f"averaging_n{n}", "averaging", n) for n in (2, 3, 5, 8, 12, 20, 40)]
    cfg += [Config(f"tie_rate{int(round(r * 100)):02d}", "tie", eps_grid[r]) for r in TIE_RATES]
    cfg += [Config(f"stability_{s}_m{m}", "stability", (s, m)) for s in ("main", "geom") for m in range(4)]
    cfg += [Config(f"randfb_q{q:g}", "random_fallback", q) for q in (0.05, 0.1, 0.2, 0.3, 0.5)]
    cfg += [Config(f"topk_ordered_k{k}", "topk", ("ordered", k, None)) for k in (1, 2, 3, 4)]
    cfg += [Config(f"topk_shuffled_k{k}", "topk", ("shuffled", k, None)) for k in (2, 3, 4)]
    cfg += [Config(f"topk_sampled_k{k}_T{T:g}", "topk", ("sampled", k, T)) for T in (0.1, 0.3) for k in (1, 2, 3, 4)]
    return cfg


def apply(cfg: Config, v: View, copies: dict[str, View], key: str):
    f = cfg.family
    if f == "argmax":
        return r_argmax(v)
    if f == "sampling":
        return r_sampling(v, cfg.param)
    if f == "averaging":
        return r_averaging(v, cfg.param)
    if f == "tie":
        return r_tie(v, cfg.param)
    if f == "stability":
        s, m = cfg.param
        names = C.PERTURB_MAIN if s == "main" else C.PERTURB_GEOM
        return r_stability(v, [copies[n] for n in names], m)
    if f == "random_fallback":
        return r_random_fallback(v, cfg.param)
    if f == "topk":
        mode, k, T = cfg.param
        return r_topk(v, k, mode, T, seed_key=key)
    raise ValueError(f)


def metric_row(d, arg_out: int, copies_d: list | None) -> np.ndarray:
    p = d.p
    first = d.p_first if d.p_first is not None else p
    pred_same = 1.0 - float(p @ p)
    if copies_d:
        pred_copy = float(np.mean([1.0 - float(p @ c.p) for c in copies_d]))
    else:
        pred_copy = np.nan
    change = 1.0 - float(p[arg_out])
    return np.r_[p, d.fb, d.tw, d.dg, d.cut, pred_same, pred_copy, change, first]


# ----------------------------------------------------------------------------
# Worker side
# ----------------------------------------------------------------------------
_G: dict = {}


def _init(split: str, layout: pd.DataFrame, dg_store: dict, configs: dict) -> None:
    _G["stores"] = {v: load_store(split, v) for v in C.VARIANTS}
    _G["layout"] = layout
    _G["dg"] = dg_store
    _G["configs"] = configs


def views_for(pid: str, aspect: float) -> dict[str, View]:
    lay = _G["layout"].loc[pid]
    W, H = int(lay.W), int(lay.H)
    out = {}
    for var in C.VARIANTS:
        s = _G["stores"][var][pid]
        f1 = map_interval((lay.f1x0, lay.f1x1), var, W)
        f2 = map_interval((lay.f2x0, lay.f2x1), var, W)
        dg = _G["dg"].get(pid) if var == "orig" else None
        out[var] = View(W, H, crop_width(H, W, aspect), f1, f2, s.xs, s.S, int(s.salient[0]), dg)
    return out


def describe(pid: str, v: View, aspect_name: str, s) -> dict:
    """Per-composite facts for the near-tie analysis (original image only)."""
    lay = _G["layout"].loc[pid]
    xs, ys, S = s.xs, s.ys, s.S

    def face_max(fx, fy):
        inx = (xs >= fx[0]) & (xs <= fx[1])
        iny = (ys >= fy[0]) & (ys <= fy[1])
        if inx.any() and iny.any():
            return float(S[np.ix_(iny, inx)].max())
        cx, cy = (fx[0] + fx[1]) / 2, (fy[0] + fy[1]) / 2
        return float(S[np.abs(ys - cy).argmin(), np.abs(xs - cx).argmin()])

    left1 = bool(lay.g1_left)
    half1 = xs < lay.w_left if left1 else xs >= lay.w_left
    sx, sy = s.salient
    in_face = lambda fx, fy: fx[0] <= sx <= fx[1] and fy[0] <= sy <= fy[1]
    f1, f2 = (lay.f1x0, lay.f1x1), (lay.f2x0, lay.f2x1)
    f1y, f2y = (lay.f1y0, lay.f1y1), (lay.f2y0, lay.f2y1)
    if in_face(f1, f1y):
        land = "face1"
    elif in_face(f2, f2y):
        land = "face2"
    elif (sx < lay.w_left) == left1:
        land = "rest1"
    else:
        land = "rest2"
    pk = v.peaks
    gap2 = float(v.m[pk[0]] - v.m[pk[1]]) if len(pk) > 1 else np.inf
    d = {f"{aspect_name}_gap2": gap2, f"{aspect_name}_npeaks": len(pk),
         f"{aspect_name}_arg_out": v.arg["out"]}
    if aspect_name == "square":
        x0, x1 = window(sx, v.crop_w, v.W)
        d.update({"s1_face": face_max(f1, f1y), "s2_face": face_max(f2, f2y),
                  "s1_half": float(S[:, half1].max()), "s2_half": float(S[:, ~half1].max()),
                  "land": land, "nx": len(xs), "ny": len(ys),
                  "top_col_matches_salient": int(np.floor(xs[pk[0]])) == int(sx),
                  "crop_matches_binary": bool(int(x0) == int(s.crops[0][0]) and int(x1 - x0) == int(s.crops[0][2]))})
    else:
        x0, x1 = window(sx, v.crop_w, v.W)
        d["portrait_crop_matches_binary"] = bool(int(x0) == int(s.crops[1][0]) and int(x1 - x0) == int(s.crops[1][2]))
    return d


def eval_one(pid: str):
    res = {}
    desc = {"pair_id": pid}
    s_orig = _G["stores"]["orig"][pid]
    for aspect_name, aspect in C.ASPECTS.items():
        views = views_for(pid, aspect)
        v = views["orig"]
        copies = {k: views[k] for k in C.VARIANTS if k != "orig"}
        arg_out = v.arg["out"]
        rows = []
        for cfg in _G["configs"][aspect_name]:
            d = apply(cfg, v, copies, pid)
            cd = None
            if cfg.family != "stability":
                cd = [apply(cfg, copies[n], {}, pid) for n in C.PERTURB_MAIN]
            rows.append(metric_row(d, arg_out, cd))
        res[aspect_name] = np.vstack(rows).astype(np.float32)
        desc.update(describe(pid, v, aspect_name, s_orig))
    return pid, res, desc


def calibrate_eps(gaps: np.ndarray) -> dict[float, float]:
    """eps values at which the given share of composites has a tie (label-free)."""
    g = np.sort(np.where(np.isfinite(gaps), gaps, np.inf))
    out = {}
    for r in TIE_RATES:
        k = int(np.ceil(r * len(g))) - 1
        out[r] = float(g[max(k, 0)]) if np.isfinite(g[max(k, 0)]) else float(np.nanmax(g[np.isfinite(g)]))
    return out


def evaluate(split: str, layout: pd.DataFrame, eps: dict[str, dict[float, float]],
             dg_store: dict | None = None, workers: int = 6) -> tuple[dict, pd.DataFrame, list[str]]:
    configs = {a: build_configs(eps[a]) for a in C.ASPECTS}
    pids = list(layout.index)
    ctx = get_context("spawn")
    out = {a: np.empty((len(pids), len(configs[a]), len(METRICS)), np.float32) for a in C.ASPECTS}
    descs = [None] * len(pids)
    pos = {p: i for i, p in enumerate(pids)}
    with ctx.Pool(workers, initializer=_init, initargs=(split, layout, dg_store or {}, configs)) as pool:
        for pid, res, desc in tqdm(pool.imap_unordered(eval_one, pids, chunksize=16), total=len(pids),
                                   desc=f"evaluate {split}"):
            i = pos[pid]
            for a in C.ASPECTS:
                out[a][i] = res[a]
            descs[i] = desc
    names = [c.name for c in configs["square"]]
    return out, pd.DataFrame(descs).set_index("pair_id"), names


def save(split: str, arrays: dict, desc: pd.DataFrame, names: list[str], eps: dict) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EVAL_DIR / f"{split}.npz", square=arrays["square"], portrait=arrays["portrait"],
                        pair_ids=np.array(desc.index), configs=np.array(names), metrics=np.array(METRICS))
    desc.to_csv(EVAL_DIR / f"{split}_desc.csv")
    (EVAL_DIR / "eps.pkl").write_bytes(pickle.dumps(eps))


def load(split: str):
    z = np.load(EVAL_DIR / f"{split}.npz", allow_pickle=True)   # written by this pipeline; pair_ids is an object array
    desc = pd.read_csv(EVAL_DIR / f"{split}_desc.csv", index_col=0)
    if "pair_ids" in z.files:
        ids = np.asarray(z["pair_ids"]).astype(str)
        if len(ids) != len(desc) or not (ids == desc.index.to_numpy().astype(str)).all():
            raise ValueError(f"{split}: rows of {split}.npz and {split}_desc.csv do not line up")
    return {"square": z["square"], "portrait": z["portrait"]}, desc, list(z["configs"]), list(z["metrics"])
