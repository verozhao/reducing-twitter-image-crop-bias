"""Where does the gender gap come from? Checks against simple photo properties.

For each test photo we measure the face box in the composite (width), the brightness
and contrast inside the face box, and whether the photo is in color. In the four
cross-gender comparisons we then ask whether women-minus-men differences in these
properties explain the gap: a weighted linear model of the argmax outcome on the
differences, whose intercept is the gap left at equal face size, brightness, contrast
and color. Intervals come from the same people-level bootstrap as the main results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from . import config as C
from .analysis import Boot, Split
from .stats import summarize

CROSS_GENDER = {"BF-BM": 1, "WF-WM": 1, "BF-WM": 1, "WF-BM": 1}   # group 1 is the woman in each


def face_properties(faces: pd.DataFrame) -> pd.DataFrame:
    """Brightness (mean luminance, 0-255) and contrast (luminance SD) inside each face box."""
    rows = []
    for r in tqdm(list(faces.itertuples()), desc="face properties"):
        img = Image.open(C.IMAGES / f"{r.qid}.jpg").convert("L")
        a = np.asarray(img, dtype=np.float32)[int(r.y1):int(r.y2), int(r.x1):int(r.x2)]
        rows.append({"qid": r.qid, "face_brightness": float(a.mean()), "face_contrast": float(a.std())})
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    test = Split("test")
    bt = Boot(test)
    ds = pd.read_csv(C.DATA / "dataset.csv").set_index("qid")
    faces = pd.read_csv(C.DATA / "faces.csv")
    faces = faces[faces.qid.isin(ds.index)]
    props = face_properties(faces).set_index("qid")
    lay = pd.read_csv(C.WORK / "composites" / "layout_test.csv", index_col=0).loc[test.desc.index]
    pairs = test.pairs

    # per-group description
    desc_rows = []
    for g in C.GROUPS:
        q = ds.index[(ds.group == g) & (ds.split == "test")]
        desc_rows.append({"group": g, "face_brightness": props.loc[q, "face_brightness"].mean(),
                          "face_contrast": props.loc[q, "face_contrast"].mean(),
                          "in_color": ds.loc[q, "in_color"].mean()})

    # women-minus-men differences per composite (cross-gender comparisons only)
    m = np.isin(test.comp, list(CROSS_GENDER))
    y = test.y_gap("square")[:, test.ci["argmax"]]        # + when the woman is kept alone
    face_w1 = (lay.f1x1 - lay.f1x0).to_numpy()
    face_w2 = (lay.f2x1 - lay.f2x0).to_numpy()
    X = np.column_stack([
        np.ones(len(y)),
        face_w1 - face_w2,
        props.loc[pairs.q1, "face_brightness"].to_numpy() - props.loc[pairs.q2, "face_brightness"].to_numpy(),
        props.loc[pairs.q1, "face_contrast"].to_numpy() - props.loc[pairs.q2, "face_contrast"].to_numpy(),
        ds.loc[pairs.q1, "in_color"].astype(float).to_numpy() - ds.loc[pairs.q2, "in_color"].astype(float).to_numpy(),
    ])[m]
    names = ["gap at equal properties", "per pixel of face width", "per unit of brightness",
             "per unit of contrast", "per color photo"]
    ym = y[m]
    Wm = bt.W[:, m]

    def fit(w):
        A = X.T @ (X * w[:, None])
        return np.linalg.solve(A, X.T @ (w * ym))

    point = fit(np.ones(len(ym)))
    boots = np.array([fit(w) for w in Wm])
    raw = summarize(ym.mean(), (Wm @ ym) / Wm.sum(axis=1))
    rows = [{"term": "gap, no adjustment", **raw}]
    for k, name in enumerate(names):
        rows.append({"term": name, **summarize(point[k], boots[:, k])})
    women_minus_men = {
        "face width (px)": float((face_w1 - face_w2)[m].mean()),
        "brightness": float(X[:, 2].mean()), "contrast": float(X[:, 3].mean()), "color photo": float(X[:, 4].mean())}
    out = pd.DataFrame(rows)
    out["mean_women_minus_men"] = out.term.map({"per pixel of face width": women_minus_men["face width (px)"],
                                                "per unit of brightness": women_minus_men["brightness"],
                                                "per unit of contrast": women_minus_men["contrast"],
                                                "per color photo": women_minus_men["color photo"]})
    res_dir = C.RESULTS / "square"
    res_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(res_dir / "gap_sources.csv", index=False)
    pd.DataFrame(desc_rows).to_csv(res_dir / "face_properties_by_group.csv", index=False)
    return out
