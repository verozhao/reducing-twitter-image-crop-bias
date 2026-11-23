import numpy as np
import pandas as pd

from cropbias import config as C
from cropbias.design import make_pairs, split_people
from cropbias.stats import composite_weights, holm, person_weights


def fake_people(n=(60, 80, 30, 50)):
    rows = []
    for g, k in zip(C.GROUPS, n):
        rows += [{"qid": f"Q{g}{i}", "group": g} for i in range(k)]
    return pd.DataFrame(rows)


def test_split_and_pairs_are_balanced():
    split = split_people(fake_people())
    assert not split.duplicated("qid").any()
    pairs = make_pairs(split, "test", 400, seed=3)
    for comp, sp in pairs.groupby("comparison"):
        assert len(sp) == 400
        assert sp.g1_left.sum() == 200                       # position balanced
        assert not sp.duplicated(["q1", "q2"]).any()          # no repeated pair
        counts = sp.q1.value_counts()
        assert counts.max() - counts.min() <= 1               # each person used about equally
        assert set(sp.q1).isdisjoint(set(split.qid[split.split == "val"]))


def test_composite_weights_use_both_people_in_every_comparison():
    """Each composite's weight must be the product of both people's counts, whatever
    order the comparison's groups come in (this once failed for WF-BF, WF-BM, WM-BM)."""
    split = split_people(fake_people())
    pairs = make_pairs(split, "test", 60, seed=5)
    ds = split[split.split == "test"]
    people = {g: np.array(sorted(ds.qid[ds.group == g])) for g in C.GROUPS}
    counts = person_weights(people, 50, 3)
    w = composite_weights(pairs, people, counts)
    idx = {g: {q: i for i, q in enumerate(people[g])} for g in C.GROUPS}
    for k, r in enumerate(pairs.itertuples()):
        expect = counts[r.g1][:, idx[r.g1][r.q1]] * counts[r.g2][:, idx[r.g2][r.q2]]
        assert np.array_equal(w[:, k], expect), (r.g1, r.g2)


def test_holm():
    adj = holm(np.array([0.01, 0.04, 0.03, 0.2]))
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2])


def test_pigeonhole_bootstrap_is_wider_than_naive_with_person_effects():
    """Crossed design with strong person effects: the people-level bootstrap SE should be
    close to the true SE across new draws of people, and the naive one too small."""
    rng = np.random.default_rng(0)
    n1, n2, M = 40, 60, 2000

    def one_dataset(seed):
        r = np.random.default_rng(seed)
        a, b = r.normal(0, 1, n1), r.normal(0, 1, n2)
        i, j = r.integers(n1, size=M), r.integers(n2, size=M)
        y = (a[i] - b[j] + r.normal(0, 1, M) > 0).astype(float)
        return i, j, y

    true_means = [one_dataset(s)[2].mean() for s in range(300)]
    true_se = np.std(true_means)
    i, j, y = one_dataset(999)
    people = {"BF": np.arange(n1), "BM": np.arange(n2), "WF": np.arange(1), "WM": np.arange(1)}
    pairs = pd.DataFrame({"g1": "BF", "q1": i, "g2": "BM", "q2": j})
    w = composite_weights(pairs, people, person_weights(people, 1000, 1))
    boot = (w @ y) / w.sum(1)
    naive = y.std() / np.sqrt(M)
    assert boot.std() > 2 * naive
    assert 0.6 * true_se < boot.std() < 1.6 * true_se
