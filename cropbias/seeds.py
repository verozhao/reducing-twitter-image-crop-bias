"""Repeat the test-pair draw with new seeds and check how much the main results move.

The main design draws its test pairs once (seed 202). Here the same test people are paired
again with nine other seeds, so there are ten draws in all. People, their validation split
and every rule setting stay fixed. Only unchanged composites are scored, so every rule
except stability fallback is covered.
"""
from __future__ import annotations

import os
import pickle
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as C
from . import evaluate as E
from .analysis import (COMP_NAMES, Boot, Split, gap_by_mix, gap_table, orientation, pooled_oriented)
from .composites import build
from .design import make_pairs
from .geometry import crop_width
from .rules import View
from .stats import summarize
from .twitter_saliency import load_store, run_binary, save_store

SEEDS = tuple(range(203, 212))     # nine new draws, plus the main draw (202)
OUT = C.RESULTS / "seeds"


def label(seed: int) -> str:
    return f"test_s{seed}"


def draw_pairs(seed: int) -> pd.DataFrame:
    split = pd.read_csv(C.DATA / "dataset.csv")
    pairs = make_pairs(split, "test", C.PAIRS_PER_COMPARISON["test"], seed)
    lab = label(seed)
    pairs["pair_id"] = pairs.pair_id.str.replace(r"^test_", lab + "_", regex=True)
    pairs["split"] = lab
    return pairs


def score_pairs(pairs: pd.DataFrame, lab: str, workers: int = 8, limit: int | None = None) -> pd.DataFrame:
    """Build each composite in memory, run Twitter's program on it, keep the grid and layout."""
    faces = pd.read_csv(C.DATA / "faces.csv").set_index("qid")
    store = load_store(lab, "orig")
    lay_path = C.WORK / "composites" / f"layout_{lab}.csv"
    lay = pd.read_csv(lay_path, index_col=0).to_dict("index") if lay_path.exists() else {}
    if limit:
        pairs = pairs[pairs.pair_id.str[-5:].astype(int) < limit]
    todo = [r for r in pairs.to_dict("records") if r["pair_id"] not in store or r["pair_id"] not in lay]
    tmpdir = C.WORK / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    local = threading.local()

    def job(r: dict):
        if not hasattr(local, "path"):
            fd, name = tempfile.mkstemp(suffix=".jpg", dir=tmpdir)
            os.close(fd)
            local.path = name
        img, ly = build(pd.Series(r), faces)
        img.save(local.path, "JPEG", quality=C.JPEG_QUALITY)
        out = run_binary(local.path)
        row = {"W": ly.W, "H": ly.H, "w_left": ly.w_left, "g1_left": r["g1_left"],
               "f1x0": ly.face1[0], "f1x1": ly.face1[1], "f2x0": ly.face2[0], "f2x1": ly.face2[1],
               "f1y0": ly.face1_y[0], "f1y1": ly.face1_y[1], "f2y0": ly.face2_y[0], "f2y1": ly.face2_y[1]}
        return r["pair_id"], out, row

    def checkpoint():
        save_store(lab, "orig", store)
        lay_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_dict(lay, orient="index").rename_axis("pair_id").to_csv(lay_path)

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(job, r) for r in todo]
        for i, fut in enumerate(tqdm(as_completed(futs), total=len(futs), desc=f"saliency {lab}")):
            pid, out, row = fut.result()
            store[pid] = out
            lay[pid] = row
            if (i + 1) % 4000 == 0:
                checkpoint()
    checkpoint()
    for f in tmpdir.glob("*.jpg"):
        f.unlink()
    return pd.DataFrame.from_dict(lay, orient="index").rename_axis("pair_id").loc[list(pairs.pair_id)]


# ----------------------------------------------------------------------------
# Evaluation on unchanged composites (square crops)
# ----------------------------------------------------------------------------
def _init(lab: str, layout: pd.DataFrame, configs: list) -> None:
    E._G["stores"] = {"orig": load_store(lab, "orig")}
    E._G["layout"] = layout
    E._G["dg"] = {}
    E._G["configs"] = configs


def _eval_one(pid: str):
    lay = E._G["layout"].loc[pid]
    s = E._G["stores"]["orig"][pid]
    W, H = int(lay.W), int(lay.H)
    v = View(W, H, crop_width(H, W, C.ASPECT_MAIN), (lay.f1x0, lay.f1x1), (lay.f2x0, lay.f2x1),
             s.xs, s.S, int(s.salient[0]), None)
    rows = [E.metric_row(E.apply(cfg, v, {}, pid), v.arg["out"], None) for cfg in E._G["configs"]]
    return pid, np.vstack(rows).astype(np.float32), {"pair_id": pid, **E.describe(pid, v, "square", s)}


def evaluate_seed(lab: str, layout: pd.DataFrame, workers: int = 6) -> None:
    eps = pickle.loads((C.WORK / "eval" / "eps.pkl").read_bytes())
    configs = [c for c in E.build_configs(eps["square"]) if c.family != "stability"]
    pids = list(layout.index)
    arr = np.empty((len(pids), len(configs), len(E.METRICS)), np.float32)
    descs = [None] * len(pids)
    pos = {p: i for i, p in enumerate(pids)}
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(lab, layout, configs)) as pool:
        for pid, res, desc in tqdm(pool.imap_unordered(_eval_one, pids, chunksize=32), total=len(pids),
                                   desc=f"evaluate {lab}"):
            arr[pos[pid]] = res
            descs[pos[pid]] = desc
    E.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(E.EVAL_DIR / f"{lab}.npz", square=arr, pair_ids=np.array(pids),
                        configs=np.array([c.name for c in configs]), metrics=np.array(E.METRICS))
    pd.DataFrame(descs).set_index("pair_id").to_csv(E.EVAL_DIR / f"{lab}_desc.csv")


# ----------------------------------------------------------------------------
# Summary of one design (the main draw or a reseeded one)
# ----------------------------------------------------------------------------
class SeedDesign(Split):
    """A Split built from a reseeded test draw (square crops only)."""

    def __init__(self, lab: str, pairs: pd.DataFrame):
        z = np.load(E.EVAL_DIR / f"{lab}.npz", allow_pickle=False)
        desc = pd.read_csv(E.EVAL_DIR / f"{lab}_desc.csv", index_col=0)
        self.name = "test"                      # same test people as the main design
        self.A = {"square": z["square"]}
        self.desc = desc
        self.pairs = pairs.set_index("pair_id").loc[desc.index]
        self.cfgs = list(z["configs"])
        self.ci = {c: i for i, c in enumerate(self.cfgs)}
        self.mi = {m: i for i, m in enumerate(z["metrics"])}
        self.comp = self.pairs.comparison.to_numpy()


def summarize_design(sp: Split, design: str, sign: dict) -> list[dict]:
    bt = Boot(sp)
    Y = sp.y_gap("square")
    Yf = sp.y_gap("square", first=True)
    arg = sp.ci["argmax"]
    rows = []

    def add(metric, s):
        rows.append({"design": design, "metric": metric, "est": s["est"], "lo": s.get("lo", np.nan),
                     "hi": s.get("hi", np.nan), "p": s.get("p", np.nan)})

    for r in gap_table(sp, bt, Y[:, [arg]], ["argmax"]).itertuples():
        add(f"argmax gap {r.comparison}", {"est": r.est, "lo": r.lo, "hi": r.hi, "p": r.p_holm})
    for r in gap_by_mix(bt, Y[:, [arg]], ["argmax"]).itertuples():
        if r.mix != "our group mix":
            add(f"argmax gap {r.comparison}, {r.mix}", {"est": r.est, "lo": r.lo, "hi": r.hi, "p": r.p})
    G_pt, G_b = pooled_oriented(bt.comp_stats(Y), sign)
    add("pooled argmax gap", summarize(G_pt[arg], G_b[:, arg]))

    arg_out = sp.desc["square_arg_out"].to_numpy()
    one = np.isin(arg_out, (1, 2)).astype(float)
    U_pt, U_b = bt.pooled(sp.metric("square", "change") * one[:, None])
    One_pt, One_b = bt.pooled(one[:, None])
    for cfg in ("tie_rate20", "tie_rate40", "sampling_T0.3", "sampling_T1", "averaging_n2"):
        j = sp.ci[cfg]
        R_pt = (G_pt[arg] - G_pt[j]) / G_pt[arg]
        R_b = (G_b[:, arg] - G_b[:, j]) / G_b[:, arg]
        u_pt, u_b = U_pt[j] / One_pt[0], U_b[:, j] / One_b[:, 0]
        add(f"{cfg} gap removed", summarize(R_pt, R_b))
        add(f"{cfg} one-person crops changed", {"est": u_pt})
        add(f"{cfg} efficiency minus random switch (2)", summarize(R_pt / u_pt - 2, R_b / u_b - 2))

    s = np.array([sign[c] for c in sp.comp])
    y = Y[:, arg] * s
    am = np.abs((sp.desc.s1_face - sp.desc.s2_face).to_numpy() * s)
    m = am <= np.quantile(am, 0.10)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (bt.W[:, m] @ y[m]) / (bt.W @ y)
    st = summarize(y[m].sum() / y.sum(), b[np.isfinite(b)])
    add("closest 10% share of gap", {**st, "p": summarize(st["est"] - 0.10, b[np.isfinite(b)] - 0.10)["p"]})

    for name, M, cfg in (("k=2 coverage gap", Y, "topk_ordered_k2"), ("k=2 first suggestion gap, best first", Yf, "topk_ordered_k2"),
                         ("k=2 first suggestion gap, shuffled", Yf, "topk_shuffled_k2")):
        g = gap_table(sp, bt, M[:, [sp.ci[cfg]]], [cfg])
        r = g[g.comparison == "gender (women - men)"].iloc[0]
        add(f"{name} (women - men)", {"est": r.est, "lo": r.lo, "hi": r.hi, "p": r.p})
    return rows


def _evaluated(lab: str, n_pairs: int) -> bool:
    """True when every pair of this seed already has its evaluation saved."""
    f = E.EVAL_DIR / f"{lab}.npz"
    return f.exists() and len(np.load(f, allow_pickle=False)["pair_ids"]) == n_pairs


def run(limit: int | None = None, workers: int = 8) -> pd.DataFrame:
    sign = orientation(Split("val"), "square")
    main = f"main (seed {C.PAIR_SEED['test']})"          # 202 unless CROPBIAS_PAIR_SEED_TEST is set
    rows = summarize_design(Split("test"), main, sign) if not limit else []
    for seed in SEEDS:
        lab = label(seed)
        pairs = draw_pairs(seed)
        if limit or not _evaluated(lab, len(pairs)):      # a finished seed is only summarized again
            layout = score_pairs(pairs, lab, workers=workers, limit=limit)
            evaluate_seed(lab, layout)
        ids = np.load(E.EVAL_DIR / f"{lab}.npz", allow_pickle=False)["pair_ids"]
        sub = pairs[pairs.pair_id.isin(ids)]
        rows += summarize_design(SeedDesign(lab, sub), f"seed {seed}", sign)
    out = pd.DataFrame(rows)
    if not limit:
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT / "seed_results.csv", index=False)
        spread = out.groupby("metric").est.agg(["mean", "min", "max", "std"]).reset_index()
        spread.to_csv(OUT / "seed_spread.csv", index=False)
    return out
