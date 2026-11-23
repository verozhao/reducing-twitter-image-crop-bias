
"""Uncertainty from resampling people, not seeds.

Every composite pairs one person from each group, and each person appears in many
pairs, so pairs are not independent. We use a two-way (pigeonhole) bootstrap
(Owen, 2007): each round redraws the people of every group with replacement, and a
composite gets weight (draws of its group-1 person) x (draws of its group-2 person).
All comparisons and all rules share the same draws, so differences between rules and
the race and gender summaries keep their dependence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def person_weights(people_by_group: dict[str, np.ndarray], iters: int, seed: int) -> dict[str, np.ndarray]:
    """For each group, an (iters, n_people) matrix of bootstrap counts."""
    rng = np.random.default_rng(seed)
    out = {}
    for g in C.GROUPS:
        n = len(people_by_group[g])
        out[g] = rng.multinomial(n, np.full(n, 1.0 / n), size=iters).astype(np.float32)
    return out


def composite_weights(pairs: pd.DataFrame, people_by_group: dict[str, np.ndarray],
                      counts: dict[str, np.ndarray]) -> np.ndarray:
    """(iters, n_composites) weights: product of the two people's bootstrap counts."""
    idx = {g: {q: i for i, q in enumerate(people_by_group[g])} for g in C.GROUPS}
    i1 = np.array([idx[g][q] for g, q in zip(pairs.g1, pairs.q1)])
    i2 = np.array([idx[g][q] for g, q in zip(pairs.g2, pairs.q2)])
    iters = next(iter(counts.values())).shape[0]

    w = np.ones((iters, len(pairs)), dtype=np.float32)

    for g in C.GROUPS:
        m1 = (pairs.g1 == g).to_numpy()
        m2 = (pairs.g2 == g).to_numpy()
        if m1.any():
            w[:, m1] *= counts[g][:, i1[m1]]
        if m2.any():
            w[:, m2] *= counts[g][:, i2[m2]]
    return w


def summarize(point: float, boot: np.ndarray) -> dict:
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided bootstrap p-value for "the value is zero"
    p = 2 * min((np.sum(boot <= 0) + 1) / (len(boot) + 1), (np.sum(boot >= 0) + 1) / (len(boot) + 1))
    return {"est": point, "lo": lo, "hi": hi, "se": float(np.std(boot, ddof=1)), "p": min(1.0, p)}


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def binomial_halfwidth(p: float, n: int) -> float:
    """95% half-width that treats every pair as independent (what the audit reported)."""
    return 1.96 * np.sqrt(p * (1 - p) / n)
