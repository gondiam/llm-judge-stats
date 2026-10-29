import math

import numpy as np
import pytest
from scipy import stats

from llm_judge_stats import (
    bootstrap_ci,
    fleiss_kappa,
    judge_adjusted_rate,
    krippendorff_alpha,
    spearman,
)

nan = math.nan
# Krippendorff (2011), "Computing Krippendorff's alpha-reliability": 4 coders x 12 units
KRIPP = np.array(
    [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
        [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan],
    ]
).T


@pytest.mark.parametrize(
    "level, expected", [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849)]
)
def test_krippendorff_published_example(level, expected):
    assert krippendorff_alpha(KRIPP, level) == pytest.approx(expected, abs=1e-3)


def test_krippendorff_labels_and_none():
    r = [["a", "a"], ["b", "b"], ["a", None], ["b", "a"]]
    assert -1 < krippendorff_alpha(r) < 1
    perfect = [["x", "x", "x"], ["y", "y", None], ["x", "x", "x"]]
    assert krippendorff_alpha(perfect) == pytest.approx(1)
    # close to Fleiss' kappa when nothing is missing (they differ by a small-sample term)
    full = np.random.default_rng(0).integers(0, 3, (200, 3))
    assert krippendorff_alpha(full) == pytest.approx(fleiss_kappa(full), abs=0.01)
    with pytest.raises(ValueError):
        krippendorff_alpha(r, "ratio")


def test_krippendorff_bootstraps_over_items():
    est, lo, hi = bootstrap_ci(krippendorff_alpha, KRIPP, n_boot=300, seed=0)
    assert lo < est < hi


def test_spearman():
    a, b = [1, 2, 3, 4, 5], [2, 1, 4, 3, 5]
    assert spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic)


def _judge(rng, truth, se, sp):
    return np.where(truth == 1, rng.random(len(truth)) < se, rng.random(len(truth)) > sp)


def test_judge_adjusted_rate_recovers_truth_and_covers():
    rng = np.random.default_rng(0)
    pi, se, sp = 0.6, 0.8, 0.95  # this judge misses passes: raw rate ~0.50
    hits_adj = hits_raw = 0
    reps = 300
    for _ in range(reps):
        truth, calib = rng.random(500) < pi, rng.random(300) < pi
        r = judge_adjusted_rate(
            _judge(rng, truth, se, sp), _judge(rng, calib, se, sp), calib, n_boot=1000, seed=rng
        )
        hits_adj += r.adjusted_ci[0] <= pi <= r.adjusted_ci[1]
        hits_raw += r.raw_ci[0] <= pi <= r.raw_ci[1]
    assert hits_adj / reps == pytest.approx(0.95, abs=0.03)
    assert hits_raw / reps < 0.05  # the uncorrected CI is confidently wrong
    w = lambda c: c[1] - c[0]
    assert w(r.raw_ci) < w(r.adjusted_ci_known_error) < w(r.adjusted_ci)
    assert "calibration items" in str(r)


def test_judge_adjusted_rate_rejects_useless_judge():
    with pytest.raises(ValueError):
        judge_adjusted_rate([1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1])  # always wrong
    with pytest.raises(ValueError):
        judge_adjusted_rate([1, 0], [1, 1], [1, 1])  # no human fails to measure specificity
