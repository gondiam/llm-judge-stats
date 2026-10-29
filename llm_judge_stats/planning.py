"""Sample-size planning: how many items to label or evaluate, decided before running."""

import math

import numpy as np
from numpy.typing import ArrayLike

from .resampling import Stat, _boot, _z


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
