import math

import numpy as np
import pytest

from llm_judge_stats import n_for_means, n_for_proportions, smallest_detectable_difference


def test_n_for_means_matches_gpower():
    # G*Power, d = 0.5, alpha = .05, power = .80
    assert n_for_means(0.5, 1, paired=False) == 64  # per group
    assert n_for_means(0.5, 1, paired=True) == 34


def _paired_rejects(rng, n, p_a, p_b, disc, reps):
    d = p_b - p_a
    cells = [(disc + d) / 2, (disc - d) / 2]  # B-only pass, A-only pass
    c = rng.multinomial(n, cells + [1 - disc], size=reps)
    b_only, a_only = c[:, 0], c[:, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (b_only - a_only) / np.sqrt(b_only + a_only)  # McNemar z
    return np.abs(np.nan_to_num(z)) > 1.96


def test_paired_proportions_power_by_simulation():
    rng = np.random.default_rng(0)
    n = n_for_proportions(0.70, 0.80, discordance=0.2)
    assert np.mean(_paired_rejects(rng, n, 0.70, 0.80, 0.2, 4000)) == pytest.approx(0.8, abs=0.03)


def test_unpaired_proportions_power_by_simulation():
    rng = np.random.default_rng(1)
    n = n_for_proportions(0.70, 0.80)
    a, b = rng.binomial(n, 0.70, 4000) / n, rng.binomial(n, 0.80, 4000) / n
    pbar = (a + b) / 2
    z = (b - a) / np.sqrt(2 * pbar * (1 - pbar) / n)
    assert np.mean(np.abs(z) > 1.96) == pytest.approx(0.8, abs=0.03)


def test_pairing_needs_fewer_items():
    assert n_for_proportions(0.7, 0.8, discordance=0.2) < n_for_proportions(0.7, 0.8)


def test_impossible_discordance_is_rejected():
    with pytest.raises(ValueError):
        n_for_proportions(0.7, 0.8, discordance=0.05)  # less than the 0.10 gap
    with pytest.raises(ValueError):
        n_for_proportions(0.95, 0.97, discordance=0.5)  # can't fit the marginals


def test_smallest_detectable_difference_inverts_planning():
    n = n_for_proportions(0.70, 0.80, discordance=0.2)
    d = smallest_detectable_difference(n, p_base=0.70, discordance=0.2)
    assert d == pytest.approx(0.10, abs=0.005) and d <= 0.10
    d = smallest_detectable_difference(n_for_means(0.3, 1.0), sd=1.0)
    assert d == pytest.approx(0.3, abs=0.01)


def test_detectability_curve_and_nan():
    curve = smallest_detectable_difference([50, 200, 800], p_base=0.7, discordance=0.2)
    assert isinstance(curve, np.ndarray) and np.all(np.diff(curve) < 0)
    assert math.isnan(smallest_detectable_difference(5, p_base=0.7, discordance=0.2))
    with pytest.raises(ValueError):
        smallest_detectable_difference(50)
