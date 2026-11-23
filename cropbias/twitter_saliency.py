"""Run Twitter's released saliency program and keep its full saliency grid.

The program prints the most salient point, one crop per requested aspect ratio, and
(with show_all_points) every grid point with its score. Rules in rules.py only use
these outputs, so the saliency model itself never changes.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from . import config as C
from .composites import composite_path, perturb

SALIENCY_DIR = C.WORK / "saliency"
ASPECT_ARGS = [C.ASPECT_MAIN, C.ASPECT_ROBUST]


@dataclass
class SalOut:
    salient: tuple[int, int]          # Twitter's most salient point (x, y)
    crops: np.ndarray                 # (n_aspects, 4) x, y, w, h from the program
    xs: np.ndarray                    # grid column centers (float32)
    ys: np.ndarray                    # grid row centers (float32)
    S: np.ndarray                     # (len(ys), len(xs)) scores (float32)


def parse(stdout: str, n_aspects: int) -> SalOut:
    lines = [l.split() for l in stdout.strip().splitlines() if l.strip()]
    salient = tuple(int(v) for v in lines[0])
    crops = np.array([[int(v) for v in l] for l in lines[1:1 + n_aspects]], dtype=np.int32)
    pts = np.array([[float(v) for v in l] for l in lines[1 + n_aspects:]], dtype=np.float64)
    if len(salient) != 2 or crops.shape != (n_aspects, 4) or pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("unexpected output from candidate_crops")
    xs = np.unique(pts[:, 0])
    ys = np.unique(pts[:, 1])
    S = np.full((len(ys), len(xs)), np.nan)
    S[np.searchsorted(ys, pts[:, 1]), np.searchsorted(xs, pts[:, 0])] = pts[:, 2]
    if np.isnan(S).any():
        raise ValueError("saliency grid has holes")
    return SalOut(salient, crops, xs.astype(np.float32), ys.astype(np.float32), S.astype(np.float32))


def run_binary(path: Path, aspects=ASPECT_ARGS) -> SalOut:
    cmd = [str(C.TWITTER_BIN), str(C.TWITTER_MODEL), str(Path(path).resolve()), "show_all_points"] + \
          [f"{a:.6f}" for a in aspects]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"candidate_crops failed ({res.returncode}): {res.stderr[:200]}")
    return parse(res.stdout, len(aspects))


def store_path(split: str, variant: str) -> Path:
    return SALIENCY_DIR / split / f"{variant}.pkl"


def load_store(split: str, variant: str) -> dict[str, SalOut]:
    p = store_path(split, variant)
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


def save_store(split: str, variant: str, store: dict[str, SalOut]) -> None:
    p = store_path(split, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)


def run_all(pair_ids: list[str], split: str, variants: list[str], workers: int = 8,
            checkpoint_every: int = 2000) -> None:
    """Score every (composite, variant). Changed copies are written to a temp file and deleted."""
    tmpdir = C.WORK / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    local = threading.local()
    for variant in variants:
        store = load_store(split, variant)
        todo = [p for p in pair_ids if p not in store]
        if not todo:
            continue

        def job(pid: str) -> tuple[str, SalOut]:
            src = composite_path(pid, split)
            if variant == "orig":
                return pid, run_binary(src)
            if not hasattr(local, "path"):
                fd, name = tempfile.mkstemp(suffix=".jpg", dir=tmpdir)
                os.close(fd)
                local.path = Path(name)
            perturb(Image.open(src).convert("RGB"), variant).save(local.path, "JPEG", quality=C.JPEG_QUALITY)
            return pid, run_binary(local.path)

        done = 0
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(job, p) for p in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"saliency {split}/{variant}"):
                pid, out = fut.result()
                store[pid] = out
                done += 1
                if done % checkpoint_every == 0:
                    save_store(split, variant, store)
        save_store(split, variant, store)
    for f in tmpdir.glob("*.jpg"):
        f.unlink()
