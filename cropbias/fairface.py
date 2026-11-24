"""FairFace (Karkkainen and Joo, WACV 2021) as a second dataset.

We use its validation images with padding 1.25, which keep the head and shoulders, and
keep adults labeled Black or White. FairFace labels are annotators' judgments of race
and gender, and its photos come from Flickr, so it differs from WikiCeleb in both the
labels and the kind of photo. The rest of the pipeline is unchanged.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from . import config as C
from .download import colorfulness

ADULT_AGES = {"20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"}
RACES = {"Black": "black", "White": "white"}


def prepare(image_root: Path, labels_csv: Path) -> pd.DataFrame:
    """Copy the selected photos into C.IMAGES and write people.csv and downloads.csv."""
    lab = pd.read_csv(labels_csv)
    sel = lab[lab.race.isin(list(RACES)) & lab.age.isin(ADULT_AGES)]
    C.IMAGES.mkdir(parents=True, exist_ok=True)
    people, downloads = [], []
    for r in tqdm(list(sel.itertuples()), desc="fairface"):
        qid = "FF_" + r.file.replace("/", "_").rsplit(".", 1)[0]
        race, gender = RACES[r.race], r.gender.lower()
        dst = C.IMAGES / f"{qid}.jpg"
        shutil.copyfile(image_root / r.file, dst)
        img = Image.open(dst).convert("RGB")
        people.append({"qid": qid, "race": race, "gender": gender, "group": C.GROUP_OF[(race, gender)],
                       "status": "ok", "age": r.age})
        downloads.append({"qid": qid, "download": "ok", "width": img.width, "height": img.height,
                          "colorfulness": colorfulness(img)})
    C.DATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(people).to_csv(C.DATA / "people.csv", index=False)
    pd.DataFrame(downloads).to_csv(C.DATA / "downloads.csv", index=False)
    return pd.DataFrame(people)
