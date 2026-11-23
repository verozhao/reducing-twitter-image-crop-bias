"""Rebuild WikiCeleb from Wikidata with the recipe of Yee et al. (2021), Section 3.1.

Twitter never released the image list, only the recipe. We run the same query today,
apply the same filters, and freeze the result in data/people.csv so later runs do not
depend on Wikidata changing.
"""
from __future__ import annotations

import html
import json
import re
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests

from . import config as C

SPARQL = """
SELECT ?human ?image ?sex_or_gender ?ethnic_group ?dob WHERE {
  ?human wdt:P31 wd:Q5 ;
         wdt:P244 ?loc_aid ;
         wdt:P18 ?image ;
         wdt:P21 ?sex_or_gender ;
         wdt:P172 ?ethnic_group .
  FILTER(?sex_or_gender IN (wd:Q6581097, wd:Q6581072))
  FILTER NOT EXISTS { ?human wdt:P106 wd:%s }
  OPTIONAL { ?human wdt:P569 ?dob }
}
""" % C.PORN_ACTOR_QID

RAW_QUERY = C.WORK_SHARED / "raw" / "wikidata_query.json"
RAW_COMMONS = C.WORK_SHARED / "raw" / "commons_metadata.json"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": C.USER_AGENT})
    return s


def run_query(path: Path = RAW_QUERY, refresh: bool = False) -> dict:
    """Run the SPARQL query once and keep the raw answer in work/raw/."""
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    r = _session().post("https://query.wikidata.org/sparql",
                        data={"query": SPARQL, "format": "json"}, timeout=180)
    r.raise_for_status()
    raw = r.json()
    raw["retrieved"] = time.strftime("%Y-%m-%d")
    path.write_text(json.dumps(raw))
    return raw


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _stable_rng(qid: str, salt: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{salt}:{qid}".encode()))


def parse_people(raw: dict) -> pd.DataFrame:
    """One row per person with all labels, then the paper's selection steps."""
    people: dict[str, dict] = defaultdict(lambda: {"genders": set(), "ethnic": set(), "images": set(), "years": set()})
    for b in raw["results"]["bindings"]:
        p = people[_qid(b["human"]["value"])]
        p["genders"].add(_qid(b["sex_or_gender"]["value"]))
        p["ethnic"].add(_qid(b["ethnic_group"]["value"]))
        p["images"].add(b["image"]["value"])
        if "dob" in b:
            m = re.match(r"^(-?\d+)-", b["dob"]["value"])
            if m:
                p["years"].add(int(m.group(1)))

    # Ethnic groups with fewer than 40 people are dropped (counted per person).
    eth_size = Counter(e for p in people.values() for e in p["ethnic"])
    big = {e for e, n in eth_size.items() if n >= C.ETHNIC_MIN_PEOPLE}

    rows = []
    for qid, p in sorted(people.items()):
        row = {"qid": qid, "birth_year": min(p["years"]) if p["years"] else np.nan,
               "ethnic_all": "|".join(sorted(p["ethnic"]))}
        mapped = sorted(e for e in p["ethnic"] if e in big and e in C.TWITTER_ETHNIC_RACE)
        races = {C.TWITTER_ETHNIC_RACE[e] for e in mapped}
        genders = {C.GENDER_QIDS[g] for g in p["genders"] if g in C.GENDER_QIDS}
        # One image per person, drawn uniformly at random (fixed per person).
        images = sorted(p["images"])
        row["image_url"] = images[_stable_rng(qid, "image").integers(len(images))]
        row["image_file"] = unquote(row["image_url"].rsplit("/", 1)[-1])
        row["n_images"] = len(images)
        row["ethnic_used"] = "|".join(mapped)
        row["race"] = races.pop() if len(races) == 1 else ""
        row["gender"] = genders.pop() if len(genders) == 1 else ""
        if not mapped:
            row["status"] = "ethnic_group_not_in_twitter_map"
        elif row["race"] == "":
            row["status"] = "labels_map_to_two_races"
        elif row["gender"] == "":
            row["status"] = "two_genders"
        else:
            row["status"] = "ok"
        row["group"] = C.GROUP_OF.get((row["race"], row["gender"]), "") if row["status"] == "ok" else ""
        if row["status"] == "ok" and row["group"] == "":
            row["status"] = "race_not_black_or_white"
        rows.append(row)
    return pd.DataFrame(rows)


def _year_from_text(text: str | None) -> float:
    if not text:
        return np.nan
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    # Any year counts, so a 1602 painting is caught too. The first year in the text is used.
    years = [int(y) for y in re.findall(r"(?<!\d)(1\d\d\d|20[0-3]\d)(?!\d)", text)]
    years = [y for y in years if 1000 <= y <= 2030]
    return float(years[0]) if years else np.nan


def commons_metadata(files: list[str], path: Path = RAW_COMMONS, batch: int = 50) -> dict[str, dict]:
    """Photo date (DateTimeOriginal), size and file type from the Commons API, cached."""
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [f for f in dict.fromkeys(files) if f not in cache]
    s = _session()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        titles = ["File:" + f for f in chunk]
        for attempt in range(8):
            r = s.get("https://commons.wikimedia.org/w/api.php", timeout=60, params={
                "action": "query", "titles": "|".join(titles), "prop": "imageinfo",
                "iiprop": "extmetadata|size|mime", "iiextmetadatafilter": "DateTimeOriginal",
                "format": "json", "formatversion": "2", "maxlag": "5"})
            if r.status_code == 200 and "error" not in r.json():
                break
            wait = float(r.headers.get("Retry-After", 0) or 0)
            time.sleep(max(wait, 10.0 * (attempt + 1)))
        r.raise_for_status()
        q = r.json()["query"]
        norm = {n["to"]: n["from"] for n in q.get("normalized", [])}
        for page in q["pages"]:
            title = norm.get(page["title"], page["title"])
            f = title[len("File:"):]
            ii = (page.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata") or {}
            em = em if isinstance(em, dict) else {}  # the API returns [] when there is no metadata
            date = (em.get("DateTimeOriginal") or {}).get("value")
            cache[f] = {"mime": ii.get("mime", ""), "width": ii.get("width"), "height": ii.get("height"),
                        "date_text": date, "missing": "imageinfo" not in page}
        for f in chunk:  # titles the API silently dropped
            cache.setdefault(f, {"mime": "", "width": None, "height": None, "date_text": None, "missing": True})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
        time.sleep(1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))
    return cache


OK_MIME = {"image/jpeg", "image/png", "image/tiff", "image/webp", "image/gif"}


def build_people_table(out_csv: Path = C.DATA / "people.csv", refresh: bool = False) -> pd.DataFrame:
    raw = run_query(refresh=refresh)
    df = parse_people(raw)
    meta = commons_metadata(df.loc[df.status == "ok", "image_file"].tolist())
    df["image_mime"] = df.image_file.map(lambda f: meta.get(f, {}).get("mime", ""))
    df["image_orig_w"] = df.image_file.map(lambda f: meta.get(f, {}).get("width"))
    df["image_orig_h"] = df.image_file.map(lambda f: meta.get(f, {}).get("height"))
    df["image_date_text"] = df.image_file.map(lambda f: (meta.get(f, {}).get("date_text") or "")[:80])
    df["image_year"] = df.image_file.map(lambda f: _year_from_text(meta.get(f, {}).get("date_text")))
    ok = df.status == "ok"
    df.loc[ok & ~df.image_mime.isin(OK_MIME), "status"] = "not_a_photo_file"
    ok = df.status == "ok"
    df.loc[ok & (df.image_year < C.PHOTO_MIN_YEAR), "status"] = "photo_before_1950"
    df["wikidata_retrieved"] = raw.get("retrieved", "")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Per-person tables (with race and gender labels) stay local in data/, which git ignores.
    # Only the counts are published, in results/people_counts.csv.
    df.to_csv(RAW_QUERY.parent / "people_all.csv", index=False)
    df[df.status != "ethnic_group_not_in_twitter_map"].to_csv(out_csv, index=False)
    counts = df.status.value_counts().rename_axis("status").reset_index(name="people")
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    counts.to_csv(C.RESULTS / "people_counts.csv", index=False)
    return df
