"""Face boxes on each source photo with MTCNN (facenet-pytorch).

The box is only used to decide who a crop keeps. Photos with no clear face, or with a
second large face, are dropped so that each photo shows exactly one person.
"""
from __future__ import annotations

from multiprocessing import get_context

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from . import config as C

_MTCNN = None


def _init_worker() -> None:
    global _MTCNN
    import torch
    from facenet_pytorch import MTCNN
    torch.set_num_threads(1)
    _MTCNN = MTCNN(keep_all=True, device="cpu", min_face_size=20, post_process=False)


def classify_faces(boxes: np.ndarray | None, probs: np.ndarray | None, img_w: int, img_h: int) -> dict:
    """Apply the one-clear-face rule to raw detections."""
    out = {"n_faces": 0, "face_status": "no_face", "x1": np.nan, "y1": np.nan, "x2": np.nan,
           "y2": np.nan, "face_prob": np.nan}
    if boxes is None or probs is None:
        return out
    keep = [(b, p) for b, p in zip(boxes, probs) if p is not None and p >= C.FACE_MIN_PROB]
    out["n_faces"] = len(keep)
    if not keep:
        return out
    keep.sort(key=lambda bp: -(bp[0][2] - bp[0][0]) * (bp[0][3] - bp[0][1]))
    (b, p) = keep[0]
    x1, y1 = max(0.0, float(b[0])), max(0.0, float(b[1]))
    x2, y2 = min(float(img_w), float(b[2])), min(float(img_h), float(b[3]))
    out.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "face_prob": float(p)})
    area0 = (b[2] - b[0]) * (b[3] - b[1])
    if len(keep) > 1:
        b2 = keep[1][0]
        if (b2[2] - b2[0]) * (b2[3] - b2[1]) >= C.FACE_SECOND_MAX_RATIO * area0:
            out["face_status"] = "multiple_faces"
            return out
    if (y2 - y1) < C.FACE_MIN_HEIGHT_FRAC * img_h:
        out["face_status"] = "face_too_small"
        return out
    out["face_status"] = "ok"
    return out


def _detect(qid: str) -> dict:
    img = Image.open(C.IMAGES / f"{qid}.jpg").convert("RGB")
    boxes, probs = _MTCNN.detect(img)
    res = classify_faces(boxes, probs, img.width, img.height)
    res.update({"qid": qid, "img_w": img.width, "img_h": img.height})
    return res


def detect_all(qids: list[str], workers: int = 4) -> pd.DataFrame:
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        rows = list(tqdm(pool.imap(_detect, qids, chunksize=8), total=len(qids), desc="faces"))
    return pd.DataFrame(rows)
