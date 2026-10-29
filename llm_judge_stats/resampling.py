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
    method: str = "percentile",
    strata: ArrayLike | None = None,
    seed: int | np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI over items: returns (estimate, lo, hi).

    `data` arrays are resampled together along axis 0 (rows = items).
    `strata` (e.g. the gold label) resamples within each group, keeping class
    balance fixed - use it for F1/MCC on imbalanced data.
    Resamples where the statistic is undefined (NaN) are dropped.

    method="percentile" is the plain percentile interval. method="bca"
    (Efron 1987) corrects for bias and skew - prefer it when the estimate sits
    near a bound (kappa or F1 near 1) or n is small. It costs n extra
    evaluations of `stat` for the jackknife.
    """
    arrays, boots = _boot(stat, data, n_boot, strata, seed)
    est = float(stat(*arrays))
    q = np.array([alpha / 2, 1 - alpha / 2])
    if method == "bca":
        nd = NormalDist()
        b = boots[~np.isnan(boots)]
        share = np.mean(b < est) + 0.5 * np.mean(b == est)
        z0 = nd.inv_cdf(min(max(share, 0.5 / len(b)), 1 - 0.5 / len(b)))
        n = len(arrays[0])
        jack = np.array([stat(*(np.delete(d, i, axis=0) for d in arrays)) for i in range(n)])
        u = np.nanmean(jack) - jack[~np.isnan(jack)]
        denom = 6 * (u**2).sum() ** 1.5
        acc = (u**3).sum() / denom if denom > 0 else 0.0
        q = np.array([nd.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))) for z in map(nd.inv_cdf, q)])
    elif method != "percentile":
        raise ValueError("method must be 'percentile' or 'bca'")
    lo, hi = np.nanpercentile(boots, 100 * q)
    return est, float(lo), float(hi)
