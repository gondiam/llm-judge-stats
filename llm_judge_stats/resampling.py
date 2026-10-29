"""Bootstrap over items, shared by the planning, comparison and agreement modules."""

from collections.abc import Callable
from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

Stat = Callable[..., Any]


def _z(alpha: float) -> float:
    return NormalDist().inv_cdf(1 - alpha / 2)


def _boot(
    stat: Stat,
    data: tuple[ArrayLike, ...],
    n_boot: int,
    strata: ArrayLike | None,
    seed: int | np.random.Generator | None,
) -> tuple[list[NDArray[Any]], NDArray[np.float64]]:
    arrays = [np.asarray(d) for d in data]
    n = len(arrays[0])
    if any(len(d) != n for d in arrays):
        raise ValueError("all arrays must have the same number of items")
    rng = np.random.default_rng(seed)
    if strata is None:
        groups = [np.arange(n)]
    else:
        s = np.asarray(strata)
        groups = [np.flatnonzero(s == v) for v in np.unique(s)]
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        out[i] = stat(*(d[idx] for d in arrays))
    return arrays, out


def bootstrap_ci(
    stat: Stat,
    *data: ArrayLike,
    n_boot: int = 2000,
    alpha: float = 0.05,
    strata: ArrayLike | None = None,
    seed: int | np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap over items: returns (estimate, lo, hi).

    `data` arrays are resampled together along axis 0 (rows = items).
    `strata` (e.g. the gold label) resamples within each group, keeping class
    balance fixed - use it for F1/MCC on imbalanced data.
    Resamples where the statistic is undefined (NaN) are dropped.
    """
    # ponytail: percentile CI; switch to BCa if estimates sit near +-1 and skew matters
    arrays, boots = _boot(stat, data, n_boot, strata, seed)
    lo, hi = np.nanpercentile(boots, [50 * alpha, 100 - 50 * alpha])
    return float(stat(*arrays)), float(lo), float(hi)
