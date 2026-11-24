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


def test_group_mix_summaries_use_the_right_weights():
    """Race summary at a given mix weights comparisons by n1 * n2; held-fixed versions average two comparisons."""
    from cropbias.analysis import AUDIT_GROUP_SIZES, COMP_NAMES, RACE_TERMS, Boot, gap_by_mix
    gaps = {"BF-BM": 0.10, "WF-BF": 0.02, "BF-WM": 0.12, "WF-BM": 0.14, "WM-BM": -0.04, "WF-WM": 0.11}
    comp = np.repeat(COMP_NAMES, 5)
    Y = np.array([gaps[c] for c in comp])[:, None]
    bt = Boot.__new__(Boot)                     # no data needed: set the fields comp_stats reads
    bt.n = {"BF": 400, "BM": 600, "WF": 100, "WM": 500}
    bt.W = np.ones((3, len(comp)))
    bt.masks = {c: comp == c for c in COMP_NAMES}
    g = gap_by_mix(bt, Y, ["argmax"]).set_index(["mix", "comparison"]).est

    def pooled(n):
        w = {c: n[c[:2]] * n[c[3:]] for c in RACE_TERMS}
        return sum(w[c] * s * gaps[c] for c, s in RACE_TERMS.items()) / sum(w.values())

    assert np.isclose(g[("our group mix", "race (White - Black)")], pooled(bt.n))
    assert np.isclose(g[("audit's group mix", "race (White - Black)")], pooled(AUDIT_GROUP_SIZES))
    assert np.isclose(g[("within gender", "race (White - Black)")], (0.02 - 0.04) / 2)
    assert np.isclose(g[("within race", "gender (women - men)")], (0.10 + 0.11) / 2)
    assert bt.n == {"BF": 400, "BM": 600, "WF": 100, "WM": 500}   # the audit mix must not overwrite the sample's sizes


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
