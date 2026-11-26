"""Command-line entry: python -m cropbias <stage> [--limit N]

Stages, in order:
  people      query Wikidata and Commons, write data/people.csv
  download    download one photo per person, write data/downloads.csv
  faces       detect faces, write data/faces.csv
  dataset     apply all filters and split people, write data/dataset.csv
  pairs       draw balanced pairs, write data/pairs.csv
  composites  build side-by-side images in work/composites/
  saliency    run Twitter's program on every composite and changed copy
  deepgaze    score a random subset of test composites with DeepGaze IIE
  evaluate    apply every rule, write work/eval/
  analyze     statistics, tables and figures in results/
  all         everything above
  seeds       (extra) redraw the test pairs with nine more seeds, results/seeds/
  fairface    (extra) prepare FairFace as a second dataset (see README)
  sources     (extra) which photo properties carry the gender gap, results/square/gap_sources.csv
--limit N keeps only the first N pairs per comparison (for a quick pilot).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config as C


def _pairs(limit: int | None) -> pd.DataFrame:
    pairs = pd.read_csv(C.DATA / "pairs.csv")
    if limit:
        pairs["k"] = pairs.pair_id.str[-5:].astype(int)
        pairs = pairs[pairs.k < limit].drop(columns="k")
    return pairs


def stage_people(args):
    from .wikidata import build_people_table
    df = build_people_table(refresh=args.refresh)
    print(df.status.value_counts().to_string())
    print(df[df.status == "ok"].group.value_counts().to_string())


def stage_download(args):
    import os
    from .download import download_all, fetch_small_originals
    people = pd.read_csv(C.DATA / "people.csv")
    # Wikimedia asks automated media downloads to keep at most 2 requests in flight.
    res = download_all(people, workers=min(2, int(os.environ.get("CROPBIAS_DL_WORKERS", 1))))
    if args.small_originals:
        small = fetch_small_originals(people)
        res = pd.concat([res[~res.qid.isin(small.qid)], small])
    res.sort_values("qid").to_csv(C.DATA / "downloads.csv", index=False)
    print(res.download.value_counts().to_string())


def stage_faces(args):
    from .faces import detect_all
    dl = pd.read_csv(C.DATA / "downloads.csv")
    qids = sorted(dl.qid[dl.download == "ok"])
    faces = detect_all(qids)
    faces.sort_values("qid").to_csv(C.DATA / "faces.csv", index=False)
    print(faces.face_status.value_counts().to_string())


def stage_dataset(args):
    from .design import split_people
    people = pd.read_csv(C.DATA / "people.csv")
    dl = pd.read_csv(C.DATA / "downloads.csv")
    faces = pd.read_csv(C.DATA / "faces.csv")
    df = people.merge(dl, on="qid", how="left").merge(faces[["qid", "face_status"]], on="qid", how="left")
    # Twitter's recipe only drops photos dated before 1950. Grey-scale photos taken later
    # stay in, as in the audit; the color score is kept so a color-only check can be run.
    steps = [("wikidata_and_commons_filters", df.status == "ok"),
             ("photo_downloaded", df.download == "ok"),
             ("one_clear_face", df.face_status == "ok")]
    keep = np.ones(len(df), bool)
    flow = []
    for name, m in steps:
        keep &= m.fillna(False).to_numpy()
        flow.append({"step": name, **{g: int((keep & (df.group == g)).sum()) for g in C.GROUPS}})
    color = keep & (df.colorfulness >= C.MIN_COLORFULNESS).to_numpy()
    flow.append({"step": "of which in color (not used as a filter)",
                 **{g: int((color & (df.group == g)).sum()) for g in C.GROUPS}})
    final = df[keep]
    small = [g for g in C.GROUPS if (final.group == g).sum() < C.SUBGROUP_MIN_PEOPLE]
    if small:
        raise SystemExit(f"groups below {C.SUBGROUP_MIN_PEOPLE} people: {small}")
    split = split_people(final)
    split = split.merge(final[["qid", "colorfulness"]], on="qid")
    split["in_color"] = split.colorfulness >= C.MIN_COLORFULNESS
    split.sort_values(["group", "qid"]).to_csv(C.DATA / "dataset.csv", index=False)
    pd.DataFrame(flow).to_csv(C.RESULTS / "dataset_flow.csv", index=False)
    print(pd.DataFrame(flow).to_string(index=False))
    print(split.groupby(["group", "split"]).size().unstack().to_string())


def stage_pairs(args):
    from .design import make_pairs
    split = pd.read_csv(C.DATA / "dataset.csv")
    pairs = pd.concat([make_pairs(split, s, C.PAIRS_PER_COMPARISON[s], C.PAIR_SEED[s]) for s in ("val", "test")])
    pairs.to_csv(C.DATA / "pairs.csv", index=False)
    print(pairs.groupby(["split", "comparison"]).size().unstack().to_string())


def stage_composites(args):
    from tqdm import tqdm
    from .composites import build, composite_path
    pairs = _pairs(args.limit)
    faces = pd.read_csv(C.DATA / "faces.csv").set_index("qid")
    for split, sp in pairs.groupby("split"):
        lay_path = C.WORK / "composites" / f"layout_{split}.csv"
        old = pd.read_csv(lay_path, index_col=0) if lay_path.exists() else pd.DataFrame()
        rows = []
        for _, pr in tqdm(sp.iterrows(), total=len(sp), desc=f"composites {split}"):
            if pr.pair_id in old.index and composite_path(pr.pair_id, split).exists():
                continue
            img, lay = build(pr, faces)
            p = composite_path(pr.pair_id, split)
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p, "JPEG", quality=C.JPEG_QUALITY)
            rows.append({"pair_id": pr.pair_id, "W": lay.W, "H": lay.H, "w_left": lay.w_left,
                         "g1_left": pr.g1_left, "f1x0": lay.face1[0], "f1x1": lay.face1[1],
                         "f2x0": lay.face2[0], "f2x1": lay.face2[1], "f1y0": lay.face1_y[0],
                         "f1y1": lay.face1_y[1], "f2y0": lay.face2_y[0], "f2y1": lay.face2_y[1]})
        new = pd.DataFrame(rows).set_index("pair_id") if rows else pd.DataFrame()
        # A rebuilt pair replaces its old layout row, so no pair is listed twice.
        old = old[~old.index.isin(new.index)]
        pd.concat([old, new]).to_csv(lay_path)


def stage_saliency(args):
    from .twitter_saliency import run_all
    pairs = _pairs(args.limit)
    for split, sp in pairs.groupby("split"):
        t = time.time()
        run_all(list(sp.pair_id), split, C.VARIANTS, workers=args.workers)
        print(f"{split}: {len(sp)} composites x {len(C.VARIANTS)} variants in {time.time() - t:.0f}s")


def deepgaze_ids(pairs: pd.DataFrame) -> list[str]:
    test = pairs[pairs.split == "test"]
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    ids = []
    for comp, sp in test.groupby("comparison"):
        ids += list(rng.permutation(sorted(sp.pair_id))[:C.DEEPGAZE_PER_COMPARISON])
    return sorted(ids)


def stage_deepgaze(args):
    from .deepgaze_quality import run
    ids = deepgaze_ids(_pairs(args.limit))
    run(ids, "test")


def stage_evaluate(args):
    import pickle
    from .evaluate import calibrate_eps, evaluate, save
    from .gaps import split_gaps
    pairs = _pairs(args.limit)
    dg_path = C.WORK / "deepgaze" / "test.pkl"
    dg = pickle.loads(dg_path.read_bytes()) if dg_path.exists() else {}
    layouts = {s: pd.read_csv(C.WORK / "composites" / f"layout_{s}.csv", index_col=0) for s in ("val", "test")}
    layouts = {s: lay.loc[lay.index.isin(pairs.pair_id[pairs.split == s])] for s, lay in layouts.items()}
    gaps = split_gaps("val", layouts["val"])
    eps = {a: calibrate_eps(gaps[a]) for a in C.ASPECTS}
    print("eps grid (calibrated on validation):", json.dumps(eps, indent=1))
    for split in ("val", "test"):
        arrays, desc, names = evaluate(split, layouts[split], eps, dg if split == "test" else {}, workers=args.workers)
        save(split, arrays, desc, names, eps)


def stage_analyze(args):
    from .analysis import run_analysis
    run_analysis()


def stage_fairface(args):
    """Prepare FairFace as a second dataset (run with CROPBIAS_RUN_DIR and CROPBIAS_IMAGES set)."""
    from .fairface import prepare
    raw = C.WORK_SHARED / "fairface_raw"
    people = prepare(raw / "images", raw / "fairface_label_val.csv")
    print(people.group.value_counts().to_string())


def stage_seeds(args):
    """Redraw the test pairs with nine more seeds and summarize how much the results move."""
    from .seeds import run
    out = run(limit=args.limit, workers=args.workers)
    print(out.pivot(index="metric", columns="design", values="est").round(3).to_string())


def stage_sources(args):
    """Fit the gender gap on face width, brightness, contrast and color (exploratory)."""
    from .mechanism import run
    print(run().round(4).to_string(index=False))


STAGES = ["people", "download", "faces", "dataset", "pairs", "composites", "saliency", "deepgaze",
          "evaluate", "analyze"]
EXTRA_STAGES = ["seeds", "fairface", "sources"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + EXTRA_STAGES + ["all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refresh", action="store_true", help="re-run the Wikidata query")
    ap.add_argument("--small-originals", action="store_true",
                    help="download: also fetch originals of 250 px or less, slowly, one at a time")
    args = ap.parse_args()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    for st in (STAGES if args.stage == "all" else [args.stage]):
        t = time.time()
        globals()[f"stage_{st}"](args)
        print(f"[{st}] done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
