"""Sample-size planning: how many items to label or evaluate, decided before running."""

import math
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from .resampling import Stat, _boot, _z

_PHI = NormalDist()


def n_for_kappa(kappa: float, prevalence: float, half_width: float, alpha: float = 0.05) -> int:
    """Items to label so a binary kappa's (1-alpha) CI has +-half_width.

    Planning formula, no pilot needed: asymptotic variance of the 2x2 kappa
    (Bloch & Kraemer 1989), assuming both raters share `prevalence`.
    """
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must be in (0, 1)")
    k, pq = kappa, prevalence * (1 - prevalence)
    var1 = (1 - k) * ((1 - k) * (1 - 2 * k) + k * (2 - k) / (2 * pq))
    return math.ceil(var1 * (_z(alpha) / half_width) ** 2)


def n_for_precision(
    stat: Stat,
    *pilot: ArrayLike,
    half_width: float,
    alpha: float = 0.05,
    n_boot: int = 2000,
    strata: ArrayLike | None = None,
    seed: int | np.random.Generator | None = None,
) -> int:
    """Items to label so `stat`'s CI has +-half_width, extrapolated from a pilot.

    Works for any statistic (kappa, F1, MCC, ICC...): bootstrap SE on the pilot,
    then n = n_pilot * (z * SE / half_width)^2 since SE shrinks as 1/sqrt(n).
    """
    data, boots = _boot(stat, pilot, n_boot, strata, seed)
    se = np.nanstd(boots, ddof=1)
    return math.ceil(len(data[0]) * (_z(alpha) * se / half_width) ** 2)


# --- comparing two variants -------------------------------------------------


def _check_paired(p_a: float, p_b: float, discordance: float) -> None:
    # discordance = share of items where exactly one variant passes. With pass
    # rates p_a, p_b that fixes the 2x2 table, which must exist.
    d, eps = p_b - p_a, 1e-9  # eps: pass rates computed as means carry float error
    b_only, a_only = (discordance + d) / 2, (discordance - d) / 2
    if not (
        abs(d) <= discordance + eps
        and discordance <= 1
        and b_only <= min(p_b, 1 - p_a) + eps
        and a_only <= min(p_a, 1 - p_b) + eps
    ):
        raise ValueError(
            f"discordance={discordance} is impossible with pass rates {p_a} and {p_b}: "
            "it must be at least |p_b - p_a| and fit both marginals"
        )


def _power_proportions(
    p_a: float, p_b: float, n: float, alpha: float, discordance: float | None
) -> float:
    d = abs(p_b - p_a)
    if discordance is None:  # two-proportion z test, n per variant
        pbar = (p_a + p_b) / 2
        s0 = math.sqrt(2 * pbar * (1 - pbar))
        s1 = math.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))
    else:  # McNemar, n paired items (Connor 1987)
        s0, s1 = math.sqrt(discordance), math.sqrt(max(discordance - d**2, 0))
    num = d * math.sqrt(n) - _z(alpha) * s0
    return _PHI.cdf(num / s1) if s1 > 0 else float(num > 0)


def n_for_proportions(
    p_a: float,
    p_b: float,
    power: float = 0.8,
    alpha: float = 0.05,
    discordance: float | None = None,
) -> int:
    """Eval items needed to tell pass rate `p_a` from `p_b` with the given power.

    Paired design (both variants run on the same items, the normal case) when
    `discordance` is given: the share of items where exactly one variant passes.
    It matters more than the pass rates - two prompts that fail on the same items
    are cheap to compare. Get it from a pilot, or try a few values. Returns the
    number of items (McNemar test, Connor 1987).

    Unpaired when `discordance` is None: returns items *per variant*
    (two-proportion z test). Both are normal approximations; below ~30 items or
    near 0/1 pass rates treat the answer as rough.
    """
    if not (0 <= p_a <= 1 and 0 <= p_b <= 1) or p_a == p_b:
        raise ValueError("p_a and p_b must be distinct pass rates in [0, 1]")
    d, z, zb = abs(p_b - p_a), _z(alpha), _PHI.inv_cdf(power)
    if discordance is None:
        pbar = (p_a + p_b) / 2
        s0 = math.sqrt(2 * pbar * (1 - pbar))
        s1 = math.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))
    else:
        _check_paired(p_a, p_b, discordance)
        s0, s1 = math.sqrt(discordance), math.sqrt(max(discordance - d**2, 0))
    return math.ceil(((z * s0 + zb * s1) / d) ** 2)


def _power_means(delta: float, sd: float, n: float, alpha: float, paired: bool) -> float:
    df = n - 1 if paired else 2 * n - 2
    nc = abs(delta) / sd * math.sqrt(n if paired else n / 2)
    t = stats.t.ppf(1 - alpha / 2, df)
    # the far tail underflows to NaN in scipy for large nc; it is ~0 there anyway
    return float(stats.nct.sf(t, df, nc) + np.nan_to_num(stats.nct.cdf(-t, df, nc)))


def n_for_means(
    delta: float, sd: float, power: float = 0.8, alpha: float = 0.05, paired: bool = True
) -> int:
    """Items needed to detect a difference `delta` in mean judge score.

    Paired (same items, both variants): `sd` is the SD of the per-item
    *differences*, usually much smaller than the SD of the scores themselves;
    returns items. Unpaired: `sd` is the score SD, assumed equal in both
    variants; returns items per variant. Exact power from the noncentral t, so
    it is right at small n too, as long as differences are roughly normal.
    """
    if delta == 0 or sd <= 0:
        raise ValueError("delta must be non-zero and sd positive")
    k = 1 if paired else 2
    n = max(2, math.ceil(k * ((_z(alpha) + _PHI.inv_cdf(power)) * sd / delta) ** 2))
    while _power_means(delta, sd, n, alpha, paired) < power:  # normal approx is a bit low
        n += 1
    return n


def smallest_detectable_difference(
    n: ArrayLike,
    *,
    p_base: float | None = None,
    discordance: float | None = None,
    sd: float | None = None,
    paired: bool = True,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float | NDArray[np.float64]:
    """The smallest difference that `n` items detect with the given power, either direction.

    The question to ask of an eval set you already have. Pass an array of n to
    get a detectability curve. Two cases, pick one:

    - pass rates: give `p_base` (the current pass rate). The answer is the
      smallest rise or drop from it that is detectable. Paired when
      `discordance` is given (assumed constant across differences, see
      `n_for_proportions`); n is per variant otherwise.
    - mean scores: give `sd`, with `paired` as in `n_for_means`.

    NaN where nothing in the possible range is detectable.
    """
    if (p_base is None) == (sd is None):
        raise ValueError("give exactly one of p_base (pass rates) or sd (mean scores)")
    if discordance is not None and not 0 < discordance <= 1:
        raise ValueError("discordance must be in (0, 1]")

    def one(m: float) -> float:
        if sd is not None:

            def gap(d: float) -> float:
                return _power_means(d, sd, m, alpha, paired) - power

            hi = sd
            while gap(hi) < 0:
                hi *= 2
            return float(optimize.brentq(gap, 1e-12 * sd, hi))
        assert p_base is not None
        # power is symmetric under swapping pass/fail, so a drop from p_base is
        # a rise from 1 - p_base; take whichever direction detects less
        both = [d for d in (up(m, p_base), up(m, 1 - p_base)) if not math.isnan(d)]
        return min(both, default=math.nan)

    def up(m: float, p0: float) -> float:
        hi = 1 - p0
        if discordance is not None:
            hi = min(hi, discordance, 2 * (1 - p0) - discordance)

        def gap_p(d: float) -> float:
            return _power_proportions(p0, p0 + d, m, alpha, discordance) - power

        if hi <= 0 or gap_p(hi) < 0:
            return math.nan
        return float(optimize.brentq(gap_p, 1e-12, hi))

    ns = np.asarray(n, float)
    out = np.array([one(m) for m in ns.ravel()]).reshape(ns.shape)
    return float(out) if out.ndim == 0 else out
