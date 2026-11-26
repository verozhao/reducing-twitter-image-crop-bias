"""Run paired preprocessing diagnostics or train/evaluate a saliency-map correction.

See MAP_CORRECTION.md for data preparation, isolation and interpretation.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from cropbias import config as C
from cropbias.map_correction import (VARIANTS, evaluate_scores, inputs, load_model,
                                     predict, preprocess_scores, train)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest="action", required=True)
    pp = subs.add_parser("preprocess", help="score changed copies and compare them to the original")
    pp.add_argument("--split", choices=("val", "test"), default="val")
    pp.add_argument("--variant", choices=VARIANTS, required=True)
    pp.add_argument("--limit", type=int, help="first N existing pairs per comparison")
    pp.add_argument("--refresh", action="store_true", help="re-score all requested copies")
    pp.add_argument("--bootstrap-iters", type=int, default=1000)
    tr = subs.add_parser("train", help="fit on validation people only, select epoch on disjoint people")
    tr.add_argument("--limit", type=int)
    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--per-group", type=int, default=8)
    tr.add_argument("--strength", type=float, default=1.0)
    tr.add_argument("--temperature", type=float, default=0.2)
    tr.add_argument("--fairness", type=float, default=1.0)
    tr.add_argument("--fidelity", type=float, default=0.1)
    tr.add_argument("--smoothness", type=float, default=0.03)
    tr.add_argument("--lr", type=float, default=0.0005)
    tr.add_argument("--seed", type=int, default=43)
    tr.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    tr.add_argument("--output", type=Path, default=C.WORK / "map_correction" / "model")
    ev = subs.add_parser("eval", help="compare corrected vs original argmax on held-out people")
    ev.add_argument("--split", choices=("val", "test"), default="test")
    ev.add_argument("--checkpoint", type=Path, default=C.WORK / "map_correction" / "model" / "best.pt")
    ev.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    ev.add_argument("--limit", type=int)
    ev.add_argument("--bootstrap-iters", type=int, default=1000)
    args = ap.parse_args()
    if getattr(args, "limit", None) is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if getattr(args, "bootstrap_iters", 1000) < 50:
        ap.error("--bootstrap-iters must be >= 50")
    if args.action == "preprocess":
        path = preprocess_scores(args.split, args.variant, args.limit, args.refresh)
        pairs, lay, original = inputs(args.split, args.limit)
        new = pickle.loads(path.read_bytes())
        target = C.WORK / "map_correction" / "preprocess_eval" / args.split / args.variant
        report = evaluate_scores(pairs, lay, original, new, args.split, target, args.bootstrap_iters)
        print(report.to_string(index=False))
        print(f"Saved {target / 'summary.csv'}")
    elif args.action == "train":
        if (args.epochs < 1 or args.per_group < 2 or args.strength <= 0 or args.temperature <= 0
                or min(args.fairness, args.fidelity, args.smoothness) < 0 or args.lr <= 0):
            ap.error("epochs>=1, per-group>=2, strength>0, temperature>0, lr>0, weights>=0 required")
        pairs, lay, original = inputs("val", args.limit)
        train(pairs, lay, original, args.output, epochs=args.epochs,
              per_group=args.per_group, strength=args.strength, temperature=args.temperature,
              fairness=args.fairness, fidelity=args.fidelity, smoothness=args.smoothness,
              lr=args.lr, seed=args.seed, device=args.device)
    else:
        pairs, lay, original = inputs(args.split, args.limit)
        model, state = load_model(args.checkpoint, args.device)
        if args.split == "val":
            fit = set(state["fit_people"])
            dev = set(state["dev_people"])
            pairs = pairs[pairs.q1.isin(dev) & pairs.q2.isin(dev)].reset_index(drop=True)
            if not pairs.empty and (set(pairs.q1) | set(pairs.q2)) & fit:
                raise RuntimeError("validation evaluation includes training person")
        if pairs.empty:
            ap.error("no person-disjoint pairs available")
        new = {}
        for i, pid in enumerate(pairs.pair_id):
            new[pid] = predict(model, pid, original[pid], args.split, state["strength"], args.device)
            if (i + 1) % 1000 == 0:
                print(f"corrected {i+1}/{len(pairs)}", flush=True)
        target = C.WORK / "map_correction" / "evaluation" / args.split
        report = evaluate_scores(pairs, lay, original, new, args.split, target, args.bootstrap_iters)
        print(report.to_string(index=False))
        print(f"Saved {target / 'summary.csv'}")


if __name__ == "__main__":
    main()
