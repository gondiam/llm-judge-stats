import math

import numpy as np
import pytest

from llm_judge_stats import (
    bootstrap_ci,
    cohen_kappa,
    f1,
    fleiss_kappa,
    icc,
    majority_vote,
    mcc,
    n_for_kappa,
    n_for_precision,
    shuffled_judge,
)


def test_cohen_kappa():
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)  # po=.75, pe=.5
    assert cohen_kappa(["a", "b"], ["a", "b"]) == 1
    assert math.isnan(cohen_kappa([1, 1], [1, 1]))


def test_fleiss_kappa_wikipedia_example():
    counts = [
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7],
    ]
    labels = [np.repeat(np.arange(5), row) for row in counts]  # 14 raters per item
    assert fleiss_kappa(labels) == pytest.approx(0.210, abs=1e-3)


def test_icc_shrout_fleiss_1979():
    x = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    assert icc(x, "1,1") == pytest.approx(0.17, abs=0.01)
    assert icc(x, "2,1") == pytest.approx(0.29, abs=0.01)
    assert icc(x, "3,1") == pytest.approx(0.71, abs=0.01)


def test_f1_mcc():
    y, p = [1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0]  # tp=2 fp=1 fn=1 tn=2
    assert f1(y, p) == pytest.approx(2 / 3)
    assert mcc(y, p) == pytest.approx(1 / 3)
    assert mcc([1, 0], [1, 1]) == 0.0


def test_stratified_bootstrap_keeps_class_balance():
    y = np.array([1] * 10 + [0] * 90)
    est, lo, hi = bootstrap_ci(np.sum, y, strata=y, n_boot=200, seed=0)
    assert est == lo == hi == 10
    est, lo, hi = bootstrap_ci(np.sum, y, n_boot=200, seed=0)
    assert lo < 10 < hi


def _simulate_pairs(rng, n, kappa, p):
    # 2x2 cell probs with shared marginal p and the given kappa
    pq = p * (1 - p)
    cells = [p * p + kappa * pq, (1 - kappa) * pq, (1 - kappa) * pq, (1 - p) ** 2 + kappa * pq]
    c = rng.choice(4, size=n, p=cells)
    return (c < 2).astype(int), (c % 2 == 0).astype(int)


def test_sample_size_matches_simulation():
    rng = np.random.default_rng(0)
    kappa, p, hw = 0.6, 0.3, 0.1
    n = n_for_kappa(kappa, p, hw)
    ks = [cohen_kappa(*_simulate_pairs(rng, n, kappa, p)) for _ in range(2000)]
    # empirical half-width of the 95% interval should be ~hw
    assert 1.96 * np.std(ks) == pytest.approx(hw, rel=0.1)
    # pilot-based extrapolation agrees with the planning formula
    pilot = _simulate_pairs(rng, 400, kappa, p)
    assert n_for_precision(cohen_kappa, *pilot, half_width=hw, seed=0) == pytest.approx(n, rel=0.25)


def test_shuffled_judge_removes_position_bias():
    rng = np.random.default_rng(0)
    first = lambda opts: 0  # always picks whatever is shown first
    picks = [shuffled_judge(first, ["a", "b"], rng) for _ in range(2000)]
    assert np.mean(picks) == pytest.approx(0.5, abs=0.05)
    best = lambda opts: opts.index(max(opts))
    assert all(shuffled_judge(best, [3, 9, 1], rng) == 1 for _ in range(50))
    assert shuffled_judge(lambda o: None, [1, 2], rng) is None


def test_majority_vote():
    assert majority_vote(["a", "b", "a"]) == "a"
    assert majority_vote(["a", "b", None]) is None
    assert majority_vote([None], tie="tie") == "tie"


def test_bca_matches_scipy():
    from scipy import stats

    x = np.random.default_rng(0).exponential(size=30)  # skewed: BCa and percentile differ
    ours = bootstrap_ci(np.mean, x, method="bca", n_boot=20000, seed=1)
    ref = stats.bootstrap((x,), np.mean, method="BCa", n_resamples=20000, random_state=2)
    assert ours[1:] == pytest.approx(
        (ref.confidence_interval.low, ref.confidence_interval.high), rel=0.03
    )
    pct = bootstrap_ci(np.mean, x, n_boot=20000, seed=1)
    assert ours[2] > pct[2]  # right skew pushes the BCa upper bound up
    with pytest.raises(ValueError):
        bootstrap_ci(np.mean, x, method="basic")
