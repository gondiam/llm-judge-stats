import numpy as np
import pytest
from scipy import stats

from llm_judge_stats import adjust_pvalues, compare, compare_many


def test_adjust_pvalues_by_hand():
    p = [0.01, 0.04, 0.03, 0.005]
    assert adjust_pvalues(p, "holm") == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert adjust_pvalues(p, "bh") == pytest.approx([0.02, 0.04, 0.04, 0.02])
    with pytest.raises(ValueError):
        adjust_pvalues(p, "bonferroni")


def test_paired_binary_is_exact_mcnemar():
    # 8 items only B passes, 2 only A passes, 30 agree
    a = np.array([0] * 8 + [1] * 2 + [1] * 20 + [0] * 10)
    b = np.array([1] * 8 + [0] * 2 + [1] * 20 + [0] * 10)
    c = compare(a, b)
    assert c.binary and c.paired and c.difference == pytest.approx(0.15)
    assert c.p_value == pytest.approx(stats.binomtest(8, 10).pvalue)
    assert c.spread == pytest.approx(10 / 40)
    assert c.ci[0] < c.difference < c.ci[1]


def test_continuous_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(5, 1, 30), rng.normal(5.5, 1, 30)
    assert compare(a, b).p_value == pytest.approx(stats.ttest_rel(b, a).pvalue)
    welch = stats.ttest_ind(b, a, equal_var=False).pvalue
    assert compare(a, b, paired=False).p_value == pytest.approx(welch)


def test_verdict_text():
    rng = np.random.default_rng(1)
    a = rng.binomial(1, 0.7, 40)
    b = a.copy()
    b[:6] = 1
    text = str(compare(a, b, min_relevant=0.05, names=("old", "new")))
    assert "new vs old" in text and "Not distinguishable" in text and "paired items" in text
    assert compare(a, b, min_relevant=0.05).n_needed is not None
    assert "no difference within reach" in str(compare(a, b))  # only 5% of items differ
    flips = a.copy()
    flips[:16] = 1 - flips[:16]
    assert "smallest difference detectable" in str(compare(a, flips))
    big = np.r_[np.zeros(50), np.ones(150)]
    assert "B is higher" in str(compare(np.zeros(200), big))


def test_input_checks():
    with pytest.raises(ValueError):
        compare([1, 0, 1], [1, 0])
    with pytest.raises(ValueError):
        compare([1, np.nan], [1, 0])


def _rates(rng, reps, alpha=0.05, **kw):
    """Type I error rate and CI coverage of `compare` on simulated null data."""
    hits = cover = 0
    for _ in range(reps):
        a, b, true_diff = kw["draw"](rng)
        c = compare(a, b, paired=kw["paired"], alpha=alpha)
        hits += c.p_value < alpha
        cover += c.ci[0] <= true_diff <= c.ci[1]
    return hits / reps, cover / reps


def _paired_binary(rng, n=40, p=0.7, disc=0.2, d=0.0):
    cells = [(disc + d) / 2, (disc - d) / 2, p - (disc - d) / 2]  # b-only, a-only, both
    c = rng.choice(4, size=n, p=cells + [1 - sum(cells)])
    return (c == 1) | (c == 2), (c == 0) | (c == 2), d


@pytest.mark.parametrize(
    "draw, paired",
    [
        (_paired_binary, True),
        (lambda r: (r.binomial(1, 0.7, 40), r.binomial(1, 0.7, 40), 0.0), False),
        (lambda r: (*(lambda x: (x, x + r.normal(0, 1, 30)))(r.normal(5, 2, 30)), 0.0), True),
        (lambda r: (r.normal(5, 1, 30), r.normal(5, 3, 30), 0.0), False),
    ],
    ids=["paired-binary", "unpaired-binary", "paired-scores", "welch"],
)
def test_type_i_error_and_coverage(draw, paired):
    type1, coverage = _rates(np.random.default_rng(2), 2000, draw=draw, paired=paired)
    assert type1 <= 0.065  # MC SE ~0.005; exact McNemar is conservative
    assert coverage == pytest.approx(0.95, abs=0.02)


def test_compare_many_adjusts_and_names():
    rng = np.random.default_rng(3)
    base = rng.binomial(1, 0.6, 100)
    scores = {"base": base, "v1": base.copy(), "v2": np.maximum(base, rng.binomial(1, 0.5, 100))}
    res = compare_many(scores, "base")
    assert [r.label for r in res] == ["v1 vs base", "v2 vs base"]
    assert all(r.p_adjusted is not None and r.p_adjusted >= r.p_value for r in res)
    assert "v2 is higher" in str(res[1]) and "Holm-adjusted" in str(res[1])
    with pytest.raises(ValueError):
        compare_many(scores, "missing")
