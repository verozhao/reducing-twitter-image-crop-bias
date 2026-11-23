"""Statistics, tables and figures from work/eval/ into results/.

Conventions
- gap: half the difference between how often only the group-1 person is kept and how
  often only the group-2 person is kept. Positive means the direction Twitter's audit found.
- Every interval is a 95% interval from the people-level bootstrap (stats.py), and every
  p-value tests "no gap" (or "no change from argmax"), Holm-adjusted over the six comparisons.
- Rule settings are fixed on validation pairs without demographic labels: each family is
  compared at the setting whose crops differ from argmax for about 10% (and 20%) of photos.
"""
from __future__ import annotations

import ast
import copy
import json
import re

import numpy as np
import pandas as pd

from . import config as C
from .evaluate import load
from .stats import binomial_halfwidth, composite_weights, holm, person_weights, summarize

COMP_NAMES = [f"{a}-{b}" for a, b in C.COMPARISONS]
RACE_TERMS = {"WF-BF": 1, "BF-WM": -1, "WF-BM": 1, "WM-BM": 1}     # + means White kept more
GENDER_TERMS = {"BF-BM": 1, "WF-WM": 1, "BF-WM": 1, "WF-BM": 1}    # + means women kept more
BUDGETS = [0.10, 0.20]
FAMILY_PREFIX = {"sampling": "sampling_", "averaging": "averaging_", "tie": "tie_rate",
                 "stability_main": "stability_main_", "stability_geom": "stability_geom_"}
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Categorical palette in its validated fixed order: blue, orange, aqua, yellow, magenta, green, violet
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
class Split:
    def __init__(self, name: str):
        arrays, desc, cfgs, metrics = load(name)
        pairs = pd.read_csv(C.DATA / "pairs.csv").set_index("pair_id")
        self.name = name
        self.A = arrays
        self.desc = desc
        self.pairs = pairs.loc[desc.index]
        self.cfgs = cfgs
        self.ci = {c: i for i, c in enumerate(cfgs)}
        self.mi = {m: i for i, m in enumerate(metrics)}
        self.comp = self.pairs.comparison.to_numpy()

    def metric(self, aspect: str, name: str) -> np.ndarray:
        """(n_composites, n_configs) array of one metric."""
        return self.A[aspect][:, :, self.mi[name]].astype(np.float64)

    def y_gap(self, aspect: str, first: bool = False) -> np.ndarray:
        a, b = ("f_only1", "f_only2") if first else ("p_only1", "p_only2")
        return 0.5 * (self.metric(aspect, a) - self.metric(aspect, b))


class Boot:
    """People-level bootstrap weights for the test split, shared by every statistic."""

    def __init__(self, sp: Split):
        ds = pd.read_csv(C.DATA / "dataset.csv")
        self.people = {g: np.array(sorted(ds.qid[(ds.group == g) & (ds.split == sp.name)])) for g in C.GROUPS}
        self.n = {g: len(v) for g, v in self.people.items()}
        counts = person_weights(self.people, C.BOOTSTRAP_ITERS, C.BOOTSTRAP_SEED)
        self.W = composite_weights(sp.pairs, self.people, counts).astype(np.float64)
        self.masks = {c: sp.comp == c for c in COMP_NAMES}

    def comp_stats(self, Y: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per comparison: point estimate (k,) and bootstrap draws (B, k) of the mean of Y."""
        out = {}
        for c, m in self.masks.items():
            Wm = self.W[:, m]
            out[c] = (Y[m].mean(axis=0), (Wm @ Y[m]) / Wm.sum(axis=1, keepdims=True))
        return out

    def overall(self, cs: dict, terms: dict[str, int]):
        """Race or gender summary: all person pairs of the two sides weighted equally."""
        w = {c: self.n[c.split("-")[0]] * self.n[c.split("-")[1]] for c in terms}
        tot = sum(w.values())
        pt = sum(w[c] * s * cs[c][0] for c, s in terms.items()) / tot
        bt = sum(w[c] * s * cs[c][1] for c, s in terms.items()) / tot
        return pt, bt

    def pooled(self, Y: np.ndarray):
        Wm = self.W
        return Y.mean(axis=0), (Wm @ Y) / Wm.sum(axis=1, keepdims=True)


def gap_table(sp: Split, bt: Boot, Y: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Gap per comparison plus race and gender summaries, for every column of Y."""
    cs = bt.comp_stats(Y)
    rows = []
    for j, name in enumerate(names):
        pvals = []
        for c in COMP_NAMES:
            s = summarize(cs[c][0][j], cs[c][1][:, j])
            rows.append({"config": name, "comparison": c, **s})
            pvals.append(s["p"])
        adj = holm(np.array(pvals))
        for k in range(len(COMP_NAMES)):
            rows[-len(COMP_NAMES) + k]["p_holm"] = adj[k]
        for label, terms in (("race (White - Black)", RACE_TERMS), ("gender (women - men)", GENDER_TERMS)):
            pt, b = bt.overall(cs, terms)
            s = summarize(pt[j], b[:, j])
            rows.append({"config": name, "comparison": label, **s, "p_holm": s["p"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Twitter's reference numbers (from their released results.txt)
# ----------------------------------------------------------------------------
def twitter_reference() -> pd.DataFrame:
    code = {("black", "female"): "BF", ("black", "male"): "BM", ("white", "female"): "WF", ("white", "male"): "WM"}
    rows = []
    for line in C.TWITTER_RESULTS.read_text().splitlines():
        m = re.search(r"'group_name_list': (.*?), 'stats': array\(\[\s*([\d.]+),\s*([\d.]+)\]\), "
                      r"'num_iterations': (\d+), 'setting': '(\w+)'", line)
        if not m:
            continue
        groups, a, b, n, setting = m.groups()
        g = ast.literal_eval(groups)
        a, b, n = float(a), float(b), int(n)
        if isinstance(g[0], tuple):
            g1, g2 = code[g[0]], code[g[1]]
            for (x, y) in C.COMPARISONS:
                if {x, y} == {g1, g2}:
                    share_x = a / n if x == g1 else b / n
                    rows.append({"setting": setting, "n": n, "comparison": f"{x}-{y}", "gap": share_x - 0.5,
                                 "binomial_halfwidth": binomial_halfwidth(share_x, n)})
        elif g == ("black", "white"):
            rows.append({"setting": setting, "n": n, "comparison": "race (White - Black)", "gap": b / n - 0.5,
                         "binomial_halfwidth": binomial_halfwidth(b / n, n)})
        elif g == ("female", "male"):
            rows.append({"setting": setting, "n": n, "comparison": "gender (women - men)", "gap": a / n - 0.5,
                         "binomial_halfwidth": binomial_halfwidth(a / n, n)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Operating points, chosen on validation without labels
# ----------------------------------------------------------------------------
def operating_points(val: Split, aspect: str) -> pd.DataFrame:
    change = val.metric(aspect, "change").mean(axis=0)
    rows = []
    for fam, prefix in FAMILY_PREFIX.items():
        idx = [i for i, c in enumerate(val.cfgs) if c.startswith(prefix)]
        for b in BUDGETS:
            j = min(idx, key=lambda i: abs(change[i] - b))
            rows.append({"family": fam, "budget": b, "config": val.cfgs[j], "val_change": change[j]})
    return pd.DataFrame(rows)


def orientation(val: Split, aspect: str) -> dict[str, float]:
    """Sign of the argmax gap per comparison on validation people (used to pool comparisons)."""
    y = val.y_gap(aspect)[:, val.ci["argmax"]]
    return {c: float(np.sign(y[val.comp == c].mean()) or 1.0) for c in COMP_NAMES}


def pooled_oriented(cs: dict, sign: dict[str, float]):
    pt = sum(sign[c] * cs[c][0] for c in COMP_NAMES) / len(COMP_NAMES)
    bt = sum(sign[c] * cs[c][1] for c in COMP_NAMES) / len(COMP_NAMES)
    return pt, bt


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def run_analysis() -> None:
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    test, val = Split("test"), Split("val")
    bt = Boot(test)
    summary: dict = {"n_test_people": bt.n, "n_test_composites": int(len(test.desc)),
                     "bootstrap_iters": C.BOOTSTRAP_ITERS}

    # Sanity checks against Twitter's program output
    d = test.desc
    summary["crop_matches_program_square"] = float(d.crop_matches_binary.mean())
    summary["crop_matches_program_portrait"] = float(d.portrait_crop_matches_binary.mean())
    summary["top_column_is_program_point"] = float(d.top_col_matches_salient.mean())

    for aspect in C.ASPECTS:
        out = C.RESULTS / aspect
        out.mkdir(exist_ok=True)
        sign = orientation(val, aspect)
        Y = test.y_gap(aspect)
        Yf = test.y_gap(aspect, first=True)
        all_gaps = gap_table(test, bt, Y, test.cfgs)
        all_gaps.to_csv(out / "gaps_all_rules.csv", index=False)

        # quality and predictability, pooled over comparisons
        qrows = []
        # How often the first suggestion (or the single crop) changes when the rule runs again
        F = np.stack([test.metric(aspect, k) for k in ("f_neither", "f_only1", "f_only2", "f_both")], axis=-1)
        pred_first = 1.0 - (F ** 2).sum(axis=-1)
        for m in ("p_neither", "p_both", "fb", "cut", "tw", "dg", "change", "pred_same", "pred_copy", "pred_first"):
            V = pred_first if m == "pred_first" else test.metric(aspect, m)
            ok = np.isfinite(V).all(axis=1) if m == "dg" else np.ones(len(V), bool)
            if m == "pred_copy":
                ok = np.isfinite(V[:, 0])
            if ok.sum() == 0:
                continue
            Wm = bt.W[:, ok]
            Vm = np.nan_to_num(V[ok])
            pt = Vm.mean(axis=0)
            b = (Wm @ Vm) / Wm.sum(axis=1, keepdims=True)
            diff_pt = pt - pt[test.ci["argmax"]]
            diff_b = b - b[:, [test.ci["argmax"]]]
            for j, name in enumerate(test.cfgs):
                if m == "pred_copy" and not np.isfinite(V[:, j]).all():
                    continue
                s = summarize(pt[j], b[:, j])
                sd = summarize(diff_pt[j], diff_b[:, j])
                qrows.append({"config": name, "metric": m, "n_composites": int(ok.sum()), **s,
                              "diff_vs_argmax": sd["est"], "diff_lo": sd["lo"], "diff_hi": sd["hi"],
                              "diff_p": sd["p"]})
        quality = pd.DataFrame(qrows)
        quality.to_csv(out / "quality_all_rules.csv", index=False)

        # rule minus argmax, per comparison (paired, same people draws)
        D = Y - Y[:, [test.ci["argmax"]]]
        dt = gap_table(test, bt, D, test.cfgs).rename(columns={"est": "diff", "p": "p_diff", "p_holm": "p_diff_holm"})
        dt.to_csv(out / "gap_change_vs_argmax.csv", index=False)

        # pooled oriented gap and the efficiency analysis (removed share vs touched share)
        cs = bt.comp_stats(Y)
        G_pt, G_b = pooled_oriented(cs, sign)
        arg = test.ci["argmax"]
        arg_out = test.desc[f"{aspect}_arg_out"].to_numpy()
        one = np.isin(arg_out, (1, 2)).astype(float)
        touched = test.metric(aspect, "change") * one[:, None]
        U_pt, U_b = bt.pooled(touched)
        One_pt, One_b = bt.pooled(one[:, None])
        # Matching random controls, both exact: showing the full photo for a random share x
        # of one-person crops removes x of the gap (efficiency 1); switching a random share x
        # to the other face reverses those outcomes and removes 2x (efficiency 2).
        # Gap among the photos a rule still crops (differs from the gap only for fallback rules)
        kept_crop = 1.0 - test.metric(aspect, "fb")
        Gc_pt = sum(sign[c] * Y[bt.masks[c]].sum(axis=0) / kept_crop[bt.masks[c]].sum(axis=0) for c in COMP_NAMES) / 6
        Gc_b = sum(sign[c] * (bt.W[:, bt.masks[c]] @ Y[bt.masks[c]]) / (bt.W[:, bt.masks[c]] @ kept_crop[bt.masks[c]])
                   for c in COMP_NAMES) / 6
        eff_rows = []
        for j, name in enumerate(test.cfgs):
            R_pt = (G_pt[arg] - G_pt[j]) / G_pt[arg]
            R_b = (G_b[:, arg] - G_b[:, j]) / G_b[:, arg]
            u_pt = U_pt[j] / One_pt[0]
            u_b = U_b[:, j] / One_b[:, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                E_pt = R_pt / u_pt if u_pt > 0 else np.nan
                E_b = R_b / u_b
            r = summarize(R_pt, R_b)
            e = summarize(E_pt, E_b) if u_pt > 0 else {"est": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
            if name.startswith(("stability_", "randfb_")):
                control, ctrl_name = 1.0, "random fallback"
            elif name.startswith(("sampling_", "averaging_", "tie_")):
                control, ctrl_name = 2.0, "random switch"
            else:
                control, ctrl_name = np.nan, ""
            p_ctrl = np.nan
            if u_pt > 0 and np.isfinite(control) and not name.startswith("randfb_"):
                p_ctrl = summarize(E_pt - control, E_b - control)["p"]
            eff_rows.append({"config": name, "pooled_gap": G_pt[j], "pooled_gap_lo": np.percentile(G_b[:, j], 2.5),
                             "pooled_gap_hi": np.percentile(G_b[:, j], 97.5),
                             "gap_removed_share": r["est"], "removed_lo": r["lo"], "removed_hi": r["hi"],
                             "one_person_crops_changed": u_pt, "efficiency": e["est"],
                             "efficiency_lo": e["lo"], "efficiency_hi": e["hi"],
                             "matching_control": ctrl_name, "control_efficiency": control,
                             "p_vs_control": p_ctrl, "pooled_gap_among_cropped": Gc_pt[j],
                             "among_cropped_lo": np.percentile(Gc_b[:, j], 2.5),
                             "among_cropped_hi": np.percentile(Gc_b[:, j], 97.5)})
        eff = pd.DataFrame(eff_rows)
        eff.to_csv(out / "efficiency.csv", index=False)

        # operating points (validation, label-free) and the main table
        ops = operating_points(val, aspect)
        ops.to_csv(out / "operating_points.csv", index=False)
        main_cfgs = ["argmax"] + list(dict.fromkeys(ops.config))
        main = all_gaps[all_gaps.config.isin(main_cfgs)].copy()
        main = main.merge(dt[["config", "comparison", "diff", "lo", "hi", "p_diff_holm"]]
                          .rename(columns={"lo": "diff_lo", "hi": "diff_hi"}), on=["config", "comparison"])
        main.to_csv(out / "main_table.csv", index=False)

        # top-k suggestions: coverage gap and first-suggestion gap
        tk = [c for c in test.cfgs if c.startswith("topk_")]
        tki = [test.ci[c] for c in tk]
        cov = gap_table(test, bt, Y[:, tki], tk).assign(measure="coverage (in at least one suggestion)")
        fst = gap_table(test, bt, Yf[:, tki], tk).assign(measure="first suggestion")
        pd.concat([cov, fst]).to_csv(out / "topk_gaps.csv", index=False)

        # fallback rate by comparison and by group (brightness vs skin tone check)
        fb = test.metric(aspect, "fb")
        stab = [c for c in test.cfgs if c.startswith("stability_")]
        rows = []
        for c in stab:
            j = test.ci[c]
            for g in C.GROUPS:
                m = (test.pairs.g1 == g).to_numpy() | (test.pairs.g2 == g).to_numpy()
                Wm = bt.W[:, m]
                b = (Wm @ fb[m, j]) / Wm.sum(axis=1)
                rows.append({"config": c, "group": g, **summarize(fb[m, j].mean(), b)})
            for race_pair in (("BF", "WF"), ("BM", "WM")):
                vals = []
                for g in race_pair:
                    m = (test.pairs.g1 == g).to_numpy() | (test.pairs.g2 == g).to_numpy()
                    Wm = bt.W[:, m]
                    vals.append(((fb[m, j]).mean(), (Wm @ fb[m, j]) / Wm.sum(axis=1)))
                rows.append({"config": c, "group": f"{race_pair[0]} minus {race_pair[1]}",
                             **summarize(vals[0][0] - vals[1][0], vals[0][1] - vals[1][1])})
            for comp in COMP_NAMES:
                m = bt.masks[comp]
                Wm = bt.W[:, m]
                rows.append({"config": c, "group": f"comparison {comp}",
                             **summarize(fb[m, j].mean(), (Wm @ fb[m, j]) / Wm.sum(axis=1))})
        pd.DataFrame(rows).to_csv(out / "fallback_by_group.csv", index=False)

        # The proposal's test: gap at equal crop quality, tie-aware against sampling (descriptive)
        if aspect == "square":
            equal_quality(test, quality, eff).to_csv(out / "equal_quality.csv", index=False)

        if aspect == "square":
            near = near_tie(test, bt, sign)
            near["deciles"].to_csv(out / "neartie_by_margin_decile.csv", index=False)
            near["shares"].to_csv(out / "neartie_gap_share.csv", index=False)
            near["scores"].to_csv(out / "face_score_shift.csv", index=False)
            near["landing"].to_csv(out / "argmax_landing.csv", index=False)
            near["curve"].to_csv(out / "neartie_concentration_curve.csv", index=False)
            rep = replication(test, bt, Y)
            rep.to_csv(C.RESULTS / "replication.csv", index=False)
            summary["near_tie_share"] = near["shares"].to_dict("records")
            make_figures(test, bt, all_gaps, quality, eff, ops, near, rep, sign)
        summary[f"{aspect}_orientation"] = sign
        summary[f"{aspect}_operating_points"] = ops.to_dict("records")

    faces_by_group().to_csv(C.RESULTS / "face_detection_by_group.csv", index=False)
    (C.RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


# ----------------------------------------------------------------------------
# Replication of Twitter's audit numbers
# ----------------------------------------------------------------------------
def replication(test: Split, bt: Boot, Y: np.ndarray) -> pd.DataFrame:
    land = test.desc.land.to_numpy()
    side = (np.isin(land, ("face1", "rest1")).astype(float) - 0.5)[:, None]   # Twitter's measure
    kept = Y[:, [test.ci["argmax"]]]
    t1 = gap_table(test, bt, side, ["side of top point (Twitter's measure)"])
    t2 = gap_table(test, bt, kept, ["face kept in square crop (our measure)"])
    # Color-only check: Twitter's recipe keeps black-and-white photos taken after 1950.
    ds = pd.read_csv(C.DATA / "dataset.csv").set_index("qid")
    color = ds.in_color.astype(bool)
    both = (test.pairs.q1.map(color) & test.pairs.q2.map(color)).to_numpy()
    bc = copy.copy(bt)
    bc.masks = {c: m & both for c, m in bt.masks.items()}
    bc.n = {g: int(((ds.group == g) & (ds.split == test.name) & color).sum()) for g in C.GROUPS}
    t3 = gap_table(test, bc, kept, ["face kept, both photos in color"])
    t3["n_composites"] = t3.comparison.map({c: int(m.sum()) for c, m in bc.masks.items()})
    # Is the color-only gap different from the all-photo gap? Same people draws for both.
    cs_all, cs_col = bt.comp_stats(kept), bc.comp_stats(kept)
    diff_rows = []
    for c in COMP_NAMES:
        diff_rows.append({"config": "color only minus all photos", "comparison": c,
                          **summarize(cs_col[c][0][0] - cs_all[c][0][0], cs_col[c][1][:, 0] - cs_all[c][1][:, 0])})
    for label, terms in (("race (White - Black)", RACE_TERMS), ("gender (women - men)", GENDER_TERMS)):
        pa, ba = bt.overall(cs_all, terms)
        pc, bcol = bc.overall(cs_col, terms)
        diff_rows.append({"config": "color only minus all photos", "comparison": label,
                          **summarize(pc[0] - pa[0], bcol[:, 0] - ba[:, 0])})
    t4 = pd.DataFrame(diff_rows)
    rep = pd.concat([t1, t2, t3, t4])
    naive = []
    for _, r in rep.iterrows():
        if r.comparison in COMP_NAMES:
            n = int((test.comp == r.comparison).sum())
            naive.append(binomial_halfwidth(0.5 + r.est, n))
        else:
            naive.append(np.nan)
    rep["naive_pairs_halfwidth"] = naive
    tw = twitter_reference()
    for setting, col in (("wiki_no_scaling_intersect", "twitter_no_scaling"),
                         ("wiki_fixed_height_intersect", "twitter_fixed_height")):
        sub = tw[(tw.setting == setting) & (tw.n == tw[tw.setting == setting].n.max())]
        rep[col] = rep.comparison.map(dict(zip(sub.comparison, sub.gap)))
    for setting, col in (("wiki_no_scaling", "twitter_no_scaling"), ("wiki_fixed_height", "twitter_fixed_height")):
        sub = tw[tw.setting == setting]
        extra = {r.comparison: r.gap for r in sub.itertuples() if r.comparison.startswith(("race", "gender"))
                 and (r.n == sub[sub.comparison == r.comparison].n.max())}
        rep[col] = rep[col].fillna(rep.comparison.map(extra))
    for col in ("twitter_no_scaling", "twitter_fixed_height"):
        rep[f"{col}_inside_our_ci"] = np.where(rep[col].isna(), np.nan, (rep[col] >= rep.lo) & (rep[col] <= rep.hi))
    return rep


def equal_quality(test: Split, quality: pd.DataFrame, eff: pd.DataFrame) -> pd.DataFrame:
    """Pooled gap of each switching rule at equal crop quality, by linear interpolation
    along its settings (argmax is the shared starting point). Descriptive, no intervals."""
    q = quality.set_index(["config", "metric"]).est
    e = eff.set_index("config").pooled_gap
    fams = {"tie-aware": "tie_rate", "sampling": "sampling_", "averaging": "averaging_"}
    rows = []
    for metric, levels in (("p_neither", (0.005, 0.008, 0.010)), ("dg", (0.625, 0.620, 0.617))):
        if ("argmax", metric) not in q.index:
            continue
        for fam, prefix in fams.items():
            cl = ["argmax"] + [c for c in test.cfgs if c.startswith(prefix)]
            xs = np.array([q[(c, metric)] for c in cl])
            ys = np.array([e[c] for c in cl])
            order = np.argsort(xs)
            for lv in levels:
                inside = xs.min() <= lv <= xs.max()
                rows.append({"quality_metric": metric, "level": lv, "family": fam,
                             "pooled_gap": float(np.interp(lv, xs[order], ys[order])) if inside else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Near-tie hypothesis
# ----------------------------------------------------------------------------
def near_tie(test: Split, bt: Boot, sign: dict[str, float]) -> dict:
    d = test.desc
    s = np.array([sign[c] for c in test.comp])
    y = test.y_gap("square")[:, test.ci["argmax"]] * s          # oriented argmax gap per composite
    margin = (d.s1_face - d.s2_face).to_numpy() * s               # oriented per-face score margin
    am = np.abs(margin)

    # local gap by decile of |margin|
    edges = np.quantile(am, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, am, side="right") - 1, 0, 9)
    rows = []
    for k in range(10):
        m = bins == k
        if m.sum() == 0:
            continue
        Wm = bt.W[:, m]
        b = (Wm @ y[m]) / Wm.sum(axis=1)
        rows.append({"decile": k + 1, "margin_lo": edges[k], "margin_hi": edges[k + 1],
                     "n": int(m.sum()), **summarize(y[m].mean(), b)})
    deciles = pd.DataFrame(rows)

    # share of the gap carried by the closest pairs, against their share of photos
    order = np.argsort(am, kind="stable")
    tot_pt = y.sum()
    tot_b = bt.W @ y
    share_fs = (0.05, 0.10, 0.20, 0.30, 0.50)
    bell_at = bell_curve(margin, test.comp, share_fs)
    share_rows = []
    for f, bell in zip(share_fs, bell_at):
        m = am <= np.quantile(am, f)
        pt = y[m].sum() / tot_pt
        with np.errstate(divide="ignore", invalid="ignore"):
            b = (bt.W[:, m] @ y[m]) / tot_b
        b = b[np.isfinite(b)]
        sm = summarize(pt - f, b - f)
        share_rows.append({"closest_share_of_photos": f, "share_of_gap": pt,
                           "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5),
                           "p_share_eq_photos": sm["p"], "bell_shift_prediction": bell})
    shares = pd.DataFrame(share_rows)

    cum = np.cumsum(y[order]) / tot_pt
    xs = np.arange(1, len(y) + 1) / len(y)
    keep = np.unique(np.r_[np.linspace(0, len(y) - 1, 400).astype(int)])
    curve = pd.DataFrame({"share_of_photos": xs[keep], "share_of_gap": cum[keep],
                          "bell_shift_prediction": bell_curve(margin, test.comp, xs[keep])})

    # is there a plain shift in face scores between groups?
    srows = []
    for c in COMP_NAMES:
        m = test.comp == c
        Wm = bt.W[:, m]
        dm = (d.s1_face - d.s2_face).to_numpy()[m]
        srows.append({"comparison": c, "measure": "mean face-score difference (group 1 - group 2)",
                      **summarize(dm.mean(), (Wm @ dm) / Wm.sum(axis=1)), "sd": dm.std()})
        win = (dm > 0).astype(float) - 0.5
        srows.append({"comparison": c, "measure": "P(group-1 face scores higher) - 0.5",
                      **summarize(win.mean(), (Wm @ win) / Wm.sum(axis=1)), "sd": np.nan})
    scores = pd.DataFrame(srows)

    land = pd.crosstab(test.comp, d.land.to_numpy(), normalize="index").reset_index().rename(columns={"row_0": "comparison"})
    land.columns = ["comparison"] + [str(c) for c in land.columns[1:]]
    return {"deciles": deciles, "shares": shares, "scores": scores, "landing": land, "curve": curve}


def bell_curve(margin: np.ndarray, comp: np.ndarray, fs, draws: int = 400_000) -> np.ndarray:
    """Share of the gap held by the closest share f of pairs (for each f in fs) if
    face-score margins were normal with each comparison's own mean and spread and the
    higher-scoring face always won."""
    rng = np.random.default_rng(0)
    fs = np.asarray(fs, dtype=float)
    idx = np.clip(np.round(fs * draws).astype(int) - 1, 0, draws - 1)
    num, den = np.zeros(len(fs)), 0.0
    for c in COMP_NAMES:
        m = margin[comp == c]
        z = rng.normal(m.mean(), m.std(), draws)
        y = np.sign(z[np.argsort(np.abs(z))]) / 2
        num += np.cumsum(y)[idx] / draws
        den += y.sum() / draws
    return num / den if den != 0 else np.full(len(fs), np.nan)


# ----------------------------------------------------------------------------
# Face detection rates by group (detector check)
# ----------------------------------------------------------------------------
def faces_by_group() -> pd.DataFrame:
    people = pd.read_csv(C.DATA / "people.csv")
    dl = pd.read_csv(C.DATA / "downloads.csv")
    faces = pd.read_csv(C.DATA / "faces.csv")
    df = people[people.status == "ok"].merge(dl, on="qid").merge(faces, on="qid", how="left")
    df = df[df.download == "ok"]
    rows = []
    for g in C.GROUPS:
        sub = df[df.group == g]
        n = len(sub)
        for st in ("ok", "no_face", "multiple_faces", "face_too_small"):
            k = int((sub.face_status == st).sum())
            p = k / n if n else np.nan
            lo, hi = wilson(k, n)
            rows.append({"group": g, "face_status": st, "count": k, "n": n, "share": p, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "legend.frameon": False,
        "lines.linewidth": 2, "axes.titleweight": "bold", "axes.titlesize": 11,
    })
    return plt


LABEL = {"BF-BM": "Black women vs Black men", "WF-BF": "White women vs Black women",
         "BF-WM": "Black women vs White men", "WF-BM": "White women vs Black men",
         "WM-BM": "White men vs Black men", "WF-WM": "White women vs White men",
         "race (White - Black)": "All: White vs Black", "gender (women - men)": "All: women vs men"}


def make_figures(test, bt, gaps, quality, eff, ops, near, rep, sign) -> None:
    plt = _style()
    fig_dir = C.RESULTS / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1. replication
    order = COMP_NAMES + ["race (White - Black)", "gender (women - men)"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ypos = np.arange(len(order))[::-1]
    for k, (measure, color, off) in enumerate((("side of top point (Twitter's measure)", SERIES[0], 0.14),
                                               ("face kept in square crop (our measure)", SERIES[1], -0.14))):
        sub = rep[rep.config == measure].set_index("comparison").loc[order]
        ax.errorbar(100 * sub.est, ypos + off, xerr=[100 * (sub.est - sub.lo), 100 * (sub.hi - sub.est)],
                    fmt="o", ms=5, color=color, ecolor=color, elinewidth=1.5, capsize=0,
                    label=f"Ours: {measure.split(' (')[0]}")
    tw = rep[rep.config == "side of top point (Twitter's measure)"].set_index("comparison").loc[order]
    ax.scatter(100 * tw.twitter_no_scaling, ypos, marker="D", s=26, color=SERIES[2], zorder=3,
               label="Twitter audit (no scaling)")
    ax.scatter(100 * tw.twitter_fixed_height, ypos, marker="s", s=24, color=SERIES[3], zorder=3,
               label="Twitter audit (fixed height)")
    ax.axvline(0, color=BASE, lw=1)
    ax.set_yticks(ypos, [LABEL[o] for o in order])
    ax.set_xlabel("Gap in points (positive = first-named group kept more)")
    ax.set_title("Argmax gaps on rebuilt data vs Twitter's audit")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_replication.png", dpi=200)
    plt.close(fig)

    # 2. near-tie test
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
    dec = near["deciles"]
    axs[0].errorbar(dec.decile, 100 * dec.est, yerr=[100 * (dec.est - dec.lo), 100 * (dec.hi - dec.est)],
                    fmt="o-", color=SERIES[0], ms=5, elinewidth=1.5, capsize=0)
    axs[0].axhline(0, color=BASE, lw=1)
    axs[0].set_xticks(range(1, 11))
    axs[0].set_xlabel("Decile of the face-score margin (1 = closest pairs)")
    axs[0].set_ylabel("Argmax gap within decile (points)")
    axs[0].set_title("Where is the gap? By score margin")
    cur = near["curve"]
    axs[1].plot(100 * cur.share_of_photos, 100 * cur.share_of_gap, color=SERIES[0], label="Observed (argmax)")
    axs[1].plot(100 * cur.share_of_photos, 100 * cur.bell_shift_prediction, color=SERIES[1],
                label="If scores were a bell-shaped shift")
    axs[1].plot([0, 100], [0, 100], color=MUTED, lw=1, label="Gap spread evenly")
    axs[1].set_xlabel("Closest pairs included (% of photos)")
    axs[1].set_ylabel("Share of the total gap (%)")
    axs[1].set_title("Do near-ties carry the gap?")
    axs[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_near_tie.png", dpi=200)
    plt.close(fig)

    # 3. (a) gap left vs how many one-person crops a rule changes; (b, c) gap vs crop quality.
    # Lines follow each rule's setting from argmax outward. Fallback rules show the full
    # image, so they appear only in (a): their cost is how often they fall back.
    q = quality.set_index(["config", "metric"]).est
    e = eff.set_index("config")
    g0 = 100 * e.pooled_gap["argmax"]

    def fam_cfgs(prefix):
        cl = [c for c in test.cfgs if c.startswith(prefix)]
        return sorted(cl, key=lambda c: e.one_person_crops_changed[c])

    fams = [("sampling", "sampling_", SERIES[0]), ("averaging", "averaging_", SERIES[1]),
            ("tie-aware (ours)", "tie_rate", SERIES[2]), ("stability, with brightness (ours)", "stability_main_", SERIES[3]),
            ("stability, no brightness (ours)", "stability_geom_", SERIES[4])]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    ax = axs[0]
    xmax = 0.0
    for name, prefix, color in fams:
        cl = fam_cfgs(prefix)
        xs = [0.0] + [100 * e.one_person_crops_changed[c] for c in cl]
        ys = [g0] + [100 * e.pooled_gap[c] for c in cl]
        xmax = max(xmax, max(xs))
        ax.plot(xs, ys, "o-", ms=4, color=color, label=name)
    xx = np.linspace(0, min(100, xmax), 50)
    ax.plot(xx, g0 * (1 - xx / 100), color=SERIES[5], label="random fallback (control)")
    xs2 = np.linspace(0, min(50, xmax), 50)
    ax.plot(xs2, g0 * (1 - 2 * xs2 / 100), color=SERIES[6], label="random switch (control)")
    ax.scatter([0], [g0], s=60, color=INK, zorder=4, label="argmax (Twitter)")
    ax.set_xlabel("One-person crops the rule changes (%)")
    ax.set_title("Gap removed per crop changed")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    for ax, (metric, xlabel, title) in zip(axs[1:], (
            ("p_neither", "Crops that keep no face (%)", "Gap vs crops with no face"),
            ("dg", "DeepGaze IIE share inside the crop (%)", "Gap vs quality from a second model"))):
        if ("argmax", metric) not in q.index:
            ax.set_visible(False)
            continue
        for name, prefix, color in fams[:3]:
            cl = fam_cfgs(prefix)
            xs = [100 * q[("argmax", metric)]] + [100 * q[(c, metric)] for c in cl]
            ys = [g0] + [100 * e.pooled_gap[c] for c in cl]
            ax.plot(xs, ys, "o-", ms=4, color=color, label=name)
        ax.scatter([100 * q[("argmax", metric)]], [g0], s=60, color=INK, zorder=4, label="argmax (Twitter)")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    for ax in axs:
        ax.axhline(0, color=BASE, lw=1)
    axs[0].set_ylabel("Pooled gap, 6 comparisons (points)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(fig_dir / "fig3_tradeoff.png", dpi=200)
    plt.close(fig)

    # 4. top-k suggestions
    tk = pd.read_csv(C.RESULTS / "square" / "topk_gaps.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, label in zip(axs, ("race (White - Black)", "gender (women - men)")):
        for (mode, measure, color, name) in (("ordered", "coverage (in at least one suggestion)", SERIES[0], "In any suggestion (top k)"),
                                             ("ordered", "first suggestion", SERIES[1], "First suggestion (top k, best first)"),
                                             ("shuffled", "first suggestion", SERIES[2], "First suggestion (shuffled)"),
                                             ("sampled", "first suggestion", SERIES[3], "First suggestion (sampled, T=0.3)")):
            rows = []
            for k in (1, 2, 3, 4):
                cfg = f"topk_{mode}_k{k}" if mode != "sampled" else f"topk_sampled_k{k}_T0.3"
                if mode == "shuffled" and k == 1:
                    cfg = "topk_ordered_k1"
                r = tk[(tk.config == cfg) & (tk.measure == measure) & (tk.comparison == label)]
                if len(r):
                    rows.append((k, 100 * r.est.iloc[0], 100 * r.lo.iloc[0], 100 * r.hi.iloc[0]))
            if rows:
                k_, est, lo, hi = map(np.array, zip(*rows))
                ax.errorbar(k_ + {"#2a78d6": -0.09, "#eb6834": -0.03, "#1baf7a": 0.03, "#eda100": 0.09}[color], est,
                            yerr=[est - lo, hi - est], fmt="o-", ms=4, color=color, elinewidth=1.2, capsize=0, label=name)
        ax.axhline(0, color=BASE, lw=1)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Number of crop suggestions k")
        ax.set_title(LABEL[label])
    axs[0].set_ylabel("Gap (points)")
    axs[1].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_topk.png", dpi=200)
    plt.close(fig)

    # 5. fallback rate by group
    fb = pd.read_csv(C.RESULTS / "square" / "fallback_by_group.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cfgs = ["stability_main_m0", "stability_geom_m0", "stability_main_m1", "stability_geom_m1"]
    names = {"stability_main_m0": "with brightness, 1+ copy changes", "stability_geom_m0": "no brightness, 1+ copy changes",
             "stability_main_m1": "with brightness, 2+ copies change", "stability_geom_m1": "no brightness, 2+ copies change"}
    width = 0.2
    for i, (cfg, color) in enumerate(zip(cfgs, SERIES[:4])):
        sub = fb[(fb.config == cfg) & fb.group.isin(C.GROUPS)].set_index("group").loc[C.GROUPS]
        x = np.arange(4) + (i - 1.5) * width
        ax.bar(x, 100 * sub.est, width=width - 0.02, color=color, label=names[cfg])
        ax.errorbar(x, 100 * sub.est, yerr=[100 * (sub.est - sub.lo), 100 * (sub.hi - sub.est)], fmt="none",
                    ecolor=INK2, elinewidth=1)
    ax.set_xticks(np.arange(4), [C.GROUP_LABELS[g] for g in C.GROUPS])
    ax.set_ylabel("Photos shown in full (%)")
    ax.set_title("How often stability fallback shows the full photo, by group")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_fallback_by_group.png", dpi=200)
    plt.close(fig)
