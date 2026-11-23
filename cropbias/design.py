"""Split people into validation and test, then draw balanced pairs for each comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def split_people(people: pd.DataFrame, val_fraction: float = C.VAL_FRACTION, seed: int = C.SPLIT_SEED) -> pd.DataFrame:
    """Random 30/70 split inside each group. People never appear in both splits."""
    rng = np.random.default_rng(seed)
    parts = []
    for g in C.GROUPS:
        ids = np.array(sorted(people.loc[people.group == g, "qid"]))
        ids = rng.permutation(ids)
        n_val = int(round(val_fraction * len(ids)))
        parts.append(pd.DataFrame({"qid": ids, "group": g,
                                   "split": ["val"] * n_val + ["test"] * (len(ids) - n_val)}))
    return pd.concat(parts, ignore_index=True)


def balanced_sequence(ids: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Use every person about equally often: n draws made of whole random permutations."""
    reps = -(-n // len(ids))
    return np.concatenate([rng.permutation(ids) for _ in range(reps)])[:n]


def _fix_duplicates(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, max_rounds: int = 200) -> np.ndarray:
    b = b.copy()
    for _ in range(max_rounds):
        seen: dict[tuple, int] = {}
        dup = []
        for i, key in enumerate(zip(a, b)):
            if key in seen:
                dup.append(i)
            else:
                seen[key] = i
        if not dup:
            return b
        for i in dup:
            j = rng.integers(len(b))
            b[i], b[j] = b[j], b[i]
    return b


def make_pairs(split_df: pd.DataFrame, split: str, n_pairs: int, seed: int) -> pd.DataFrame:
    """For each comparison draw n_pairs (person from group 1, person from group 2).

    Each person appears about equally often, no pair repeats, and exactly half of the
    pairs put the group-1 person on the left (in random order), so position is balanced.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g1, g2 in C.COMPARISONS:
        p1 = np.array(sorted(split_df.qid[(split_df.group == g1) & (split_df.split == split)]))
        p2 = np.array(sorted(split_df.qid[(split_df.group == g2) & (split_df.split == split)]))
        a = balanced_sequence(p1, n_pairs, rng)
        b = balanced_sequence(p2, n_pairs, rng)
        b = _fix_duplicates(a, b, rng)
        left = rng.permutation(np.r_[np.ones(n_pairs // 2, bool), np.zeros(n_pairs - n_pairs // 2, bool)])
        comp = f"{g1}-{g2}"
        for k in range(n_pairs):
            rows.append({"pair_id": f"{split}_{g1}{g2}_{k:05d}", "split": split, "comparison": comp,
                         "g1": g1, "g2": g2, "q1": a[k], "q2": b[k], "g1_left": bool(left[k])})
    return pd.DataFrame(rows)
