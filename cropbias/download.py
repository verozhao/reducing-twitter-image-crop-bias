"""Download one Commons thumbnail per person and flag near grey-scale photos."""
from __future__ import annotations

import hashlib
import os
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from . import config as C

IMAGE_DIR = C.IMAGES


def colorfulness(img: Image.Image) -> float:
    """Hasler and Suesstrunk (2003) colorfulness. Grey-scale is 0, most color photos are above 20."""
    a = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32)
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, img).convert("RGB")
    return img.convert("RGB")


# Wikimedia serves thumbnails in these standard widths and rate-limits downloads of
# original files (https://w.wiki/GHai). We ask for the largest standard width that is
# still smaller than the original, so the request is always a thumbnail. Originals of
# 250 px or less are fetched one at a time by fetch_small_originals().
STANDARD_WIDTHS = (250, 330, 500)


def thumb_width(orig_w, max_w: int = C.DOWNLOAD_WIDTH) -> int | None:
    if orig_w is None or not np.isfinite(orig_w):
        return max_w
    ok = [w for w in STANDARD_WIDTHS if w < orig_w and w <= max_w]
    return max(ok) if ok else None


def direct_thumb_url(filename: str, width: int, mime: str = "") -> str:
    """Thumbnail address on upload.wikimedia.org. This skips the Special:FilePath redirect
    page, which is rate-limited to about one request per second."""
    name = filename.replace(" ", "_")
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    base = f"https://upload.wikimedia.org/wikipedia/commons/thumb/{h[0]}/{h[:2]}/{quote(name)}"
    if mime == "image/tiff":
        return f"{base}/lossy-page1-{width}px-{quote(name)}.jpg"
    return f"{base}/{width}px-{quote(name)}"


def _log_error(r: requests.Response) -> None:
    (C.WORK / "logs").mkdir(parents=True, exist_ok=True)
    with open(C.WORK / "logs" / "download_errors.log", "a") as log:
        log.write(f"{time.strftime('%H:%M:%S')}\t{r.status_code}\t{r.headers.get('Retry-After')}\t{r.url[:120]}\n")


class Pacer:
    """One shared pace for all threads. A 429 makes every thread wait as long as the
    server asks and halves the request rate; each success raises the rate a little,
    up to the starting rate."""

    def __init__(self, per_second: float, min_per_second: float = 0.05):
        self.max_rate = per_second
        self.min_rate = min_per_second
        self.rate = per_second
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_time)
            self.next_time = start + 1.0 / self.rate
        time.sleep(max(0.0, start - time.monotonic()))

    def success(self) -> None:
        with self.lock:
            self.rate = min(self.max_rate, self.rate * 1.05)

    def back_off(self, seconds: float) -> None:
        with self.lock:
            self.rate = max(self.min_rate, self.rate / 2)
            self.next_time = max(self.next_time, time.monotonic() + seconds)


PACER = Pacer(per_second=float(os.environ.get("CROPBIAS_DL_RATE", 1.0)))


def _fetch(session: requests.Session, filename: str, out: Path, width: int, mime: str = "") -> dict:
    urls = [direct_thumb_url(filename, width, mime),
            f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(filename)}?width={width}"]
    err = "not_tried"
    for url in urls:
        for attempt in range(10):
            PACER.wait()
            try:
                r = session.get(url, timeout=60)
            except requests.RequestException as e:
                err = type(e).__name__
                PACER.back_off(2 ** attempt)
                continue
            if r.status_code == 200:
                try:
                    img = _to_rgb(Image.open(io.BytesIO(r.content)))
                except Exception as e:  # not a readable image
                    return {"download": f"unreadable:{type(e).__name__}"}
                img.save(out, "JPEG", quality=C.JPEG_QUALITY)
                PACER.success()
                return {"download": "ok", "width": img.width, "height": img.height,
                        "colorfulness": colorfulness(img), "requested_width": width}
            err = f"http_{r.status_code}"
            _log_error(r)
            if r.status_code in (429, 500, 502, 503, 504):
                retry = float(r.headers.get("Retry-After") or 0)
                PACER.back_off(min(180.0, max(retry, 2.0 * 2 ** attempt)))
                continue
            break  # e.g. 404 on the direct address: try the redirect page instead
    return {"download": err, "requested_width": width}


def download_all(people: pd.DataFrame, width: int = C.DOWNLOAD_WIDTH, workers: int = 1) -> pd.DataFrame:
    """Download photos for people with status ok. Existing files are reused.

    Order: the smallest group first, then everyone else in a fixed random order, so a
    download cut short by rate limits is still a random sample of each group."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    todo = people[people.status == "ok"].copy()
    sizes = todo.group.value_counts()
    todo["first"] = todo.group == sizes.idxmin()
    todo["rand"] = np.random.default_rng(0).random(len(todo))
    todo = todo.sort_values(["first", "rand"], ascending=[False, True])
    session = requests.Session()
    session.headers.update({"User-Agent": C.USER_AGENT})
    results: dict[str, dict] = {}

    def job(qid: str, filename: str, orig_w, mime: str) -> tuple[str, dict]:
        out = IMAGE_DIR / f"{qid}.jpg"
        if out.exists():
            img = Image.open(out)
            return qid, {"download": "ok", "width": img.width, "height": img.height, "colorfulness": colorfulness(img)}
        w = thumb_width(orig_w, width)
        if w is None:
            return qid, {"download": "original_too_small"}
        res = _fetch(session, filename, out, w, mime)
        time.sleep(0.1)
        return qid, res

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(job, q, f, w, m) for q, f, w, m in
                zip(todo.qid, todo.image_file, todo.image_orig_w, todo.image_mime.fillna(""))]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="download"):
            qid, res = fut.result()
            results[qid] = res
    res = pd.DataFrame.from_dict(results, orient="index")
    res.index.name = "qid"
    return res.reset_index()


def fetch_small_originals(people: pd.DataFrame, pause: float = 3.0) -> pd.DataFrame:
    """Originals that are too small for a thumbnail, fetched slowly one by one."""
    session = requests.Session()
    session.headers.update({"User-Agent": C.USER_AGENT})
    rows = []
    small = people[(people.status == "ok") & people.image_orig_w.map(lambda w: thumb_width(w) is None)]
    for qid, f in tqdm(list(zip(small.qid, small.image_file)), desc="small originals"):
        out = IMAGE_DIR / f"{qid}.jpg"
        if out.exists():
            img = Image.open(out)
            rows.append({"qid": qid, "download": "ok", "width": img.width, "height": img.height,
                         "colorfulness": colorfulness(img), "requested_width": 0})
            continue
        name = f.replace(" ", "_")
        h = hashlib.md5(name.encode("utf-8")).hexdigest()
        url = f"https://upload.wikimedia.org/wikipedia/commons/{h[0]}/{h[:2]}/{quote(name)}"
        r = session.get(url, timeout=60)
        if r.status_code == 200:
            try:
                img = _to_rgb(Image.open(io.BytesIO(r.content)))
                img.save(out, "JPEG", quality=C.JPEG_QUALITY)
                rows.append({"qid": qid, "download": "ok", "width": img.width, "height": img.height,
                             "colorfulness": colorfulness(img), "requested_width": 0})
            except Exception as e:
                rows.append({"qid": qid, "download": f"unreadable:{type(e).__name__}", "requested_width": 0})
        else:
            rows.append({"qid": qid, "download": f"http_{r.status_code}", "requested_width": 0})
            if r.status_code == 429:
                time.sleep(min(600.0, float(r.headers.get("Retry-After", 60) or 60)))
        time.sleep(pause)
    return pd.DataFrame(rows)
