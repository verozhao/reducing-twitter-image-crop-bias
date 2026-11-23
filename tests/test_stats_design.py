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


def test_bootstrap_weights_wf_bf():
    # One person from each demographic group
    people = {
        g: np.array([f"Q{g}"])
        for g in C.GROUPS
    }

    # Set bootstrap counts:
    # WF appears twice, BF appears three times
    counts = {
        g: np.ones((1, 1), dtype=np.float32)
        for g in C.GROUPS
    }
    counts["WF"][0, 0] = 2
    counts["BF"][0, 0] = 3

    # One composite: White Female vs Black Female
    pairs = pd.DataFrame([{
        "g1": "WF",
        "q1": "QWF",
        "g2": "BF",
        "q2": "QBF",
    }])

    # Calculate composite bootstrap weight
    weights = composite_weights(pairs, people, counts)

    # Expected: 2 * 3 = 6
    np.testing.assert_array_equal(
        weights,
        np.array([[6.0]], dtype=np.float32)
    )

