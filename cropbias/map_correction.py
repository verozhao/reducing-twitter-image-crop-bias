"""Experimental input diagnostics and learned correction of Twitter's saliency map.

Original Twitter cropping geometry is retained; see MAP_CORRECTION.md.
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from . import config as C
from .composites import composite_path
from .geometry import crop_width, outcome, window
from .twitter_saliency import load_store, run_binary

SIZE = (128, 32)  # PIL width, height. Fixed grid for batched correction training.
VARIANTS = ("gray", "matched_luminance")


def inputs(split: str, limit: int | None = None):
    pairs = pd.read_csv(C.DATA / "pairs.csv").query("split == @split").copy()
    if limit is not None:
        pairs = pairs[pairs.pair_id.str.rsplit("_", n=1).str[-1].astype(int) < limit]
    if pairs.empty:
        raise ValueError("no pairs available")
    layout = pd.read_csv(C.WORK / "composites" / f"layout_{split}.csv", index_col="pair_id")
    missing = set(pairs.pair_id) - set(layout.index)
    if missing:
        raise ValueError(f"missing layout for {len(missing)} pairs; run composites stage")
    original = load_store(split, "orig")
    missing = set(pairs.pair_id) - set(original)
    if missing:
        raise ValueError(f"missing Twitter grid for {len(missing)} pairs; run saliency stage")
    return pairs.reset_index(drop=True), layout, original


def transform(img: Image.Image, seam: int, kind: str) -> Image.Image:
    """Apply the same transform to both source photos; dimensions stay unchanged."""
    img = img.convert("RGB")
    if kind == "gray":
        return ImageOps.grayscale(img).convert("RGB")
    if kind != "matched_luminance":
        raise ValueError(kind)
    rgb = np.asarray(img, dtype=np.float32)
    halves = [rgb[:, :seam], rgb[:, seam:]]
    values = [np.sum(h * [0.2126, 0.7152, 0.0722], axis=2) for h in halves]
    means = [float(v.mean()) for v in values]
    stds = [float(v.std()) for v in values]
    target_mean = sum(means) / 2
    target_std = sum(stds) / 2
    result = []
    for half, luminance, mean, std in zip(halves, values, means, stds):
        new_luminance = (luminance - mean) * (target_std / max(std, 1.0)) + target_mean
        result.append(np.clip(half + (new_luminance - luminance)[..., None], 0, 255).astype(np.uint8))
    return Image.fromarray(np.concatenate(result, axis=1))


def preprocess_scores(split: str, kind: str, limit: int | None = None, refresh: bool = False):
    if kind not in VARIANTS:
        raise ValueError(f"select one of {VARIANTS}")
    pairs, layout, _ = inputs(split, limit)
    target = C.WORK / "map_correction" / "preprocess" / split / f"{kind}.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    scores = pickle.loads(target.read_bytes()) if target.exists() else {}
    for i, pid in enumerate(pairs.pair_id):
        if pid in scores and not refresh:
            continue
        img = Image.open(composite_path(pid, split)).convert("RGB")
        im = transform(img, int(layout.loc[pid, "w_left"]), kind)
        with tempfile.TemporaryDirectory(dir=C.WORK) as tmp:
            path = Path(tmp) / "changed.jpg"
            im.save(path, "JPEG", quality=C.JPEG_QUALITY)
            scores[pid] = run_binary(path, aspects=[C.ASPECT_MAIN])
        if (i + 1) % 200 == 0:
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(scores, protocol=pickle.HIGHEST_PROTOCOL))
            tmp.replace(target)
            print(f"{kind} {split}: {i+1}/{len(pairs)}", flush=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(scores, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(target)
    return target


def crop_stats(lay, sal, score: np.ndarray | None = None) -> dict:
    """Use Twitter's maximum point and original square geometry, with group-1/2 face boxes."""
    W, H = int(lay.W), int(lay.H)
    grid = sal.S if score is None else score
    k = int(np.argmax(grid))
    row, col = np.unravel_index(k, grid.shape)
    x = float(sal.salient[0]) if score is None else float(sal.xs[col])
    x0, x1 = window(x, crop_width(H, W, C.ASPECT_MAIN), W)
    code = int(outcome((lay.f1x0, lay.f1x1), (lay.f2x0, lay.f2x1), x0, x1))
    # Twitter's logits are converted to a density by softmax, just as existing rules do.
    weights = np.exp(sal.S.astype(np.float64) - np.max(sal.S))
    mass = weights.sum(axis=0)
    kept = mass[(sal.xs >= x0) & (sal.xs < x1)].sum() / mass.sum()
    return {"outcome": code, "gap": 0.5 * (code == 1) - 0.5 * (code == 2),
            "tw_retained": float(kept), "x": x, "y": float(sal.ys[row])}


def summarise(df: pd.DataFrame, pairs: pd.DataFrame, split: str, iters: int = 1000):
    """Paired two-way person bootstrap; differences use the exact same pairs and draws."""
    from .stats import person_weights
    dataset = pd.read_csv(C.DATA / "dataset.csv")
    people = {}
    for g in C.GROUPS:
        available = set(dataset.qid[(dataset.group == g) & (dataset.split == split)])
        observed = set(pairs.q1[pairs.g1 == g]) | set(pairs.q2[pairs.g2 == g])
        if not observed or not observed <= available:
            raise ValueError(f"pair identities for {g} do not match {split} dataset")
        people[g] = np.array(sorted(observed))
    counts = person_weights(people, iters, C.BOOTSTRAP_SEED)
    out = []
    for comp, ids in pairs.groupby("comparison", sort=False).indices.items():
        sub = df.iloc[ids]
        sub_pairs = pairs.iloc[ids]
        g1, g2 = comp.split("-")
        ix1 = pd.Index(people[g1]).get_indexer(sub_pairs.q1)
        ix2 = pd.Index(people[g2]).get_indexer(sub_pairs.q2)
        if (ix1 < 0).any() or (ix2 < 0).any():
            raise ValueError("pair has person missing from split dataset")
        w = counts[g1][:, ix1].astype(np.float64) * counts[g2][:, ix2].astype(np.float64)
        den = w.sum(axis=1)
        def stat(key):
            vals = sub[key].to_numpy(dtype=float)
            boot = np.divide(w @ vals, den, out=np.full(len(den), np.nan), where=den > 0)
            good = boot[np.isfinite(boot)]
            lo, hi = np.percentile(good, [2.5, 97.5]) if len(good) else (np.nan, np.nan)
            return float(np.mean(vals)), float(lo), float(hi)
        a, _, _ = stat("baseline_gap")
        b, _, _ = stat("new_gap")
        delta, low, high = stat("gap_change")
        retention, _, _ = stat("tw_retained")
        base_dg = sub.baseline_deepgaze_retained.dropna()
        new_dg = sub.new_deepgaze_retained.dropna()
        out.append({"comparison": comp, "pairs": len(sub), "argmax_gap_points": 100*a,
                    "new_gap_points": 100*b, "change_points": 100*delta,
                    "change_low": 100*low, "change_high": 100*high,
                    "changed_outcome_pct": 100*float((sub.baseline_outcome != sub.new_outcome).mean()),
                    "no_face_pct": 100*float((sub.new_outcome == 0).mean()),
                    "twitter_saliency_retained_pct": 100*retention,
                    "deepgaze_pairs": len(new_dg),
                    "argmax_deepgaze_retained_pct": 100*float(base_dg.mean()) if len(base_dg) else np.nan,
                    "new_deepgaze_retained_pct": 100*float(new_dg.mean()) if len(new_dg) else np.nan})
    return pd.DataFrame(out)


def original_retention_at(lay, sal, center_x):
    W, H = int(lay.W), int(lay.H)
    x0, x1 = window(center_x, crop_width(H, W, C.ASPECT_MAIN), W)
    weights = np.exp(sal.S.astype(np.float64) - np.max(sal.S))
    mass = weights.sum(axis=0)
    return float(mass[(sal.xs >= x0) & (sal.xs < x1)].sum() / mass.sum())


def evaluate_scores(pairs, layout, original, new, split: str, output: Path, iters=1000):
    dg_path = C.WORK / "deepgaze" / "test.pkl"
    dg = pickle.loads(dg_path.read_bytes()) if split == "test" and dg_path.exists() else {}
    rows = []
    for pid in pairs.pair_id:
        if pid not in new:
            raise ValueError(f"missing altered saliency for {pid}")
        before = crop_stats(layout.loc[pid], original[pid])
        obj = new[pid]
        after = crop_stats(layout.loc[pid], obj if not isinstance(obj, np.ndarray) else original[pid],
                           obj if isinstance(obj, np.ndarray) else None)
        density = dg.get(pid)
        def dg_retained(x):
            if density is None:
                return np.nan
            left, right = window(x, crop_width(int(layout.loc[pid, "H"]),
                                                 int(layout.loc[pid, "W"]), C.ASPECT_MAIN),
                                 int(layout.loc[pid, "W"]))
            return float(np.asarray(density)[int(left):int(right)].sum())
        rows.append({"pair_id": pid, "baseline_outcome": before["outcome"],
                     "new_outcome": after["outcome"], "baseline_gap": before["gap"],
                     "new_gap": after["gap"], "gap_change": after["gap"] - before["gap"],
                     "tw_retained": original_retention_at(layout.loc[pid], original[pid], after["x"]),
                     "new_x": after["x"],
                     "baseline_deepgaze_retained": dg_retained(before["x"]),
                     "new_deepgaze_retained": dg_retained(after["x"])})
    df = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "per_pair.csv", index=False)
    summary = summarise(df, pairs, split, iters)
    summary.to_csv(output / "summary.csv", index=False)
    return summary


def person_partition(pairs: pd.DataFrame, dataset: pd.DataFrame, seed: int = 43):
    """Reserve half of each val group for fitting and half for tuning; discard cross pairs."""
    fit = set()
    dev = set()
    rng = np.random.default_rng(seed)
    for group in C.GROUPS:
        ids = np.array(sorted(dataset.qid[(dataset.group == group) & (dataset.split == "val")]))
        ids = rng.permutation(ids)
        if len(ids) < 4:
            raise ValueError(f"too few validation people in {group}")
        fit.update(ids[:len(ids)//2])
        dev.update(ids[len(ids)//2:])
    fit_rows = pairs[pairs.q1.isin(fit) & pairs.q2.isin(fit)].copy()
    dev_rows = pairs[pairs.q1.isin(dev) & pairs.q2.isin(dev)].copy()
    for part in (fit_rows, dev_rows):
        if set(part.comparison) != {f"{a}-{b}" for a, b in C.COMPARISONS}:
            raise ValueError("each split needs every comparison; increase validation pairs")
    return fit_rows, dev_rows


def feature(pid, sal, split):
    import torch
    import torch.nn.functional as F
    image = Image.open(composite_path(pid, split)).convert("RGB").resize(SIZE, Image.Resampling.BILINEAR)
    rgb = torch.from_numpy(np.asarray(image).copy()).float().permute(2, 0, 1) / 255.0
    raw = torch.tensor(sal.S.copy(), dtype=torch.float32)[None, None]
    raw = F.interpolate(raw, size=(SIZE[1], SIZE[0]), mode="bilinear", align_corners=False)[0, 0]
    normal = ((raw - raw.mean()) / (raw.std().clamp_min(0.1))).clamp(-4, 4)
    return torch.cat((rgb, normal[None]), dim=0), raw


def keep_mask(lay, width: int = SIZE[0]):
    """Each model-grid column selects the corresponding original x crop center."""
    import torch
    xs = (np.arange(width) + 0.5) * float(lay.W) / width
    x0, x1 = window(xs, crop_width(int(lay.H), int(lay.W), C.ASPECT_MAIN), int(lay.W))
    codes = outcome((lay.f1x0, lay.f1x1), (lay.f2x0, lay.f2x1), x0, x1)
    return torch.tensor((codes == 1).astype(np.float32) - (codes == 2).astype(np.float32))


def correction_model():
    import torch
    from torch import nn
    class Correction(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(4, 16, 5, padding=2), nn.ReLU(),
                                     nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 1, 1), nn.Tanh())
            # Zero initial residual preserves the baseline exactly at step 0.
            nn.init.zeros_(self.net[-2].weight)
            nn.init.zeros_(self.net[-2].bias)
        def forward(self, x, strength):
            return self.net(x)[:, 0] * strength
    return Correction()


def _batch_examples(pairs, layout, original, split, device):
    import torch
    for comp, sub in pairs.groupby("comparison"):
        xs, ss, masks = [], [], []
        for pid in sub.pair_id:
            x, raw = feature(pid, original[pid], split)
            xs.append(x)
            ss.append(raw)
            masks.append(keep_mask(layout.loc[pid]))
        yield comp, (torch.stack(xs).to(device), torch.stack(ss).to(device),
                     torch.stack(masks).to(device))


def _batch_losses(model, X, S, M, strength, temperature):
    import torch
    delta = model(X, strength)
    logits = S + delta
    probs = torch.softmax((logits / temperature).flatten(1), dim=1).view_as(logits)
    signed = (probs.sum(dim=1) * M).sum(dim=1)
    fidelity = (delta.square()).mean()
    # Regularizer discourages highly irregular maps; the zero correction has zero cost.
    smooth = (delta[:, :, 1:] - delta[:, :, :-1]).square().mean()
    return signed, fidelity, smooth


def train(pairs, layout, original, out: Path, *, epochs=12, per_group=8,
          strength=1.0, temperature=0.2, fairness=1.0, fidelity=0.1,
          smoothness=0.03, lr=0.0005, seed=43, device="cpu"):
    import torch
    torch.manual_seed(seed)
    dataset = pd.read_csv(C.DATA / "dataset.csv")
    fit, dev = person_partition(pairs, dataset, seed)
    model = correction_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    training = dict(_batch_examples(fit, layout, original, "val", device))
    tuning = dict(_batch_examples(dev, layout, original, "val", device))
    rng = np.random.default_rng(seed)
    best = float("inf")
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        n_steps = max(1, max(len(batch[0]) for batch in training.values()) // per_group)
        for _ in range(n_steps):
            signed_means, penalties = [], []
            for X, S, M in training.values():
                ids = rng.choice(len(X), size=per_group, replace=len(X)<per_group)
                ii = torch.as_tensor(ids, dtype=torch.long, device=X.device)
                signed, fid, smooth = _batch_losses(model, X.index_select(0, ii),
                                                     S.index_select(0, ii), M.index_select(0, ii),
                                                     strength, temperature)
                signed_means.append(signed.mean())
                penalties.append(fidelity*fid + smoothness*smooth)
            loss = fairness * torch.stack(signed_means).square().mean() + torch.stack(penalties).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            dev_gaps, dev_fid = [], []
            for X, S, M in tuning.values():
                signed, fid, _ = _batch_losses(model, X, S, M, strength, temperature)
                dev_gaps.append(float(signed.mean()))
                dev_fid.append(float(fid))
            # Model selection is based ONLY on held-out validation people.
            criterion = float(np.mean(np.abs(dev_gaps)) + fidelity*np.mean(dev_fid))
        history.append({"epoch": epoch, "dev_objective": criterion,
                        "dev_mean_abs_soft_gap": float(np.mean(np.abs(dev_gaps))),
                        "dev_correction_mse": float(np.mean(dev_fid))})
        print(history[-1], flush=True)
        if criterion < best:
            best = criterion
            torch.save({"state_dict": model.state_dict(), "strength": strength,
                        "temperature": temperature, "seed": seed,
                        "fit_people": sorted(set(fit.q1) | set(fit.q2)),
                        "dev_people": sorted(set(dev.q1) | set(dev.q2)),
                        "epoch": epoch}, out / "best.pt")
    pd.DataFrame(history).to_csv(out / "train_history.csv", index=False)
    print(f"Saved {out / 'best.pt'}; fit pairs={len(fit)}; dev pairs={len(dev)}")


def predict(model, pid, sal, split, strength, device):
    import torch
    import torch.nn.functional as F
    x, _ = feature(pid, sal, split)
    with torch.no_grad():
        delta = model(x[None].to(device), strength)[:, None]
        delta = F.interpolate(delta, size=sal.S.shape, mode="bilinear", align_corners=False)[0, 0]
    return sal.S + delta.cpu().numpy()


def load_model(checkpoint: Path, device="cpu"):
    import torch
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = correction_model().to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state
