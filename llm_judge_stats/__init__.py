"""Statistical tools for evaluating LLM judges against humans (or each other)."""

import math
from collections import Counter
from statistics import NormalDist

import numpy as np

__all__ = [
    "cohen_kappa",
    "fleiss_kappa",
    "icc",
    "f1",
    "mcc",
    "bootstrap_ci",
    "n_for_kappa",
    "n_for_precision",
    "shuffled_judge",
    "majority_vote",
]
__version__ = "0.1.0"


def _z(alpha):
    return NormalDist().inv_cdf(1 - alpha / 2)


# --- agreement -------------------------------------------------------------


def cohen_kappa(a, b):
    """Cohen's kappa between two raters' labels (any hashable categories)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("a and b must be 1-D and the same length")
    cats = np.union1d(a, b)
    pe = ((a[:, None] == cats).mean(0) * (b[:, None] == cats).mean(0)).sum()
    return float((np.mean(a == b) - pe) / (1 - pe)) if pe < 1 else math.nan


def fleiss_kappa(ratings):
    """Fleiss' kappa. `ratings` is (n_items, n_raters) of labels, no missing values."""
    r = np.asarray(ratings)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("ratings must be (n_items, n_raters) with n_raters >= 2")
    n, m = r.shape
    counts = (r[:, :, None] == np.unique(r)).sum(1)  # (n_items, n_categories)
    p_item = ((counts**2).sum(1) - m) / (m * (m - 1))
    pe = ((counts.sum(0) / (n * m)) ** 2).sum()
    return float((p_item.mean() - pe) / (1 - pe)) if pe < 1 else math.nan


def icc(scores, kind="2,1"):
    """Shrout & Fleiss (1979) single-rater ICC. `scores` is (n_items, n_raters).

    kind: "1,1" one-way random, "2,1" two-way random absolute agreement
    (the usual choice for judge-vs-human scores), "3,1" two-way mixed consistency.
    """
    x = np.asarray(scores, float)
    if x.ndim != 2 or min(x.shape) < 2:
        raise ValueError("scores must be (n_items, n_raters) with both >= 2")
    n, k = x.shape
    gm = x.mean()
    ssr = k * ((x.mean(1) - gm) ** 2).sum()
    ssc = n * ((x.mean(0) - gm) ** 2).sum()
    sse = ((x - gm) ** 2).sum() - ssr - ssc
    msr, msc = ssr / (n - 1), ssc / (k - 1)
    mse, msw = sse / ((n - 1) * (k - 1)), (ssc + sse) / (n * (k - 1))
    if kind == "1,1":
        return float((msr - msw) / (msr + (k - 1) * msw))
    if kind == "2,1":
        return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))
    if kind == "3,1":
        return float((msr - mse) / (msr + (k - 1) * mse))
    raise ValueError("kind must be '1,1', '2,1' or '3,1'")


# --- classification vs. gold ---------------------------------------------


def _confusion(y_true, y_pred, positive):
    t, p = np.asarray(y_true) == positive, np.asarray(y_pred) == positive
    return [float(v) for v in ((t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum())]


def f1(y_true, y_pred, positive=1):
    """Binary F1 for `positive`; NaN when there are no positives at all."""
    tp, fp, fn, _ = _confusion(y_true, y_pred, positive)
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else math.nan


def mcc(y_true, y_pred, positive=1):
    """Binary Matthews correlation; 0 when a marginal is empty (sklearn convention)."""
    tp, fp, fn, tn = _confusion(y_true, y_pred, positive)
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


# --- uncertainty and sample size ------------------------------------------


def _boot(stat, data, n_boot, strata, seed):
    data = [np.asarray(d) for d in data]
    n = len(data[0])
    if any(len(d) != n for d in data):
        raise ValueError("all arrays must have the same number of items")
    rng = np.random.default_rng(seed)
    if strata is None:
        groups = [np.arange(n)]
    else:
        strata = np.asarray(strata)
        groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        out[i] = stat(*(d[idx] for d in data))
    return data, out


def bootstrap_ci(stat, *data, n_boot=2000, alpha=0.05, strata=None, seed=None):
    """Percentile bootstrap over items: returns (estimate, lo, hi).

    `data` arrays are resampled together along axis 0 (rows = items).
    `strata` (e.g. the gold label) resamples within each group, keeping class
    balance fixed - use it for F1/MCC on imbalanced data.
    Resamples where the statistic is undefined (NaN) are dropped.
    """
    # ponytail: percentile CI; switch to BCa if estimates sit near +-1 and skew matters
    data, boots = _boot(stat, data, n_boot, strata, seed)
    lo, hi = np.nanpercentile(boots, [50 * alpha, 100 - 50 * alpha])
    return float(stat(*data)), float(lo), float(hi)


def n_for_kappa(kappa, prevalence, half_width, alpha=0.05):
    """Items to label so a binary kappa's (1-alpha) CI has +-half_width.

    Planning formula, no pilot needed: asymptotic variance of the 2x2 kappa
    (Bloch & Kraemer 1989), assuming both raters share `prevalence`.
    """
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must be in (0, 1)")
    k, pq = kappa, prevalence * (1 - prevalence)
    var1 = (1 - k) * ((1 - k) * (1 - 2 * k) + k * (2 - k) / (2 * pq))
    return math.ceil(var1 * (_z(alpha) / half_width) ** 2)


def n_for_precision(stat, *pilot, half_width, alpha=0.05, n_boot=2000, strata=None, seed=None):
    """Items to label so `stat`'s CI has +-half_width, extrapolated from a pilot.

    Works for any statistic (kappa, F1, MCC, ICC...): bootstrap SE on the pilot,
    then n = n_pilot * (z * SE / half_width)^2 since SE shrinks as 1/sqrt(n).
    """
    data, boots = _boot(stat, pilot, n_boot, strata, seed)
    se = np.nanstd(boots, ddof=1)
    return math.ceil(len(data[0]) * (_z(alpha) * se / half_width) ** 2)


# --- running judges -------------------------------------------------------


def shuffled_judge(judge, options, rng=None):
    """Show `options` to `judge` in random order to neutralise position bias.

    `judge(list) -> index into that list` (or None for a tie). The pick is
    mapped back to an index into the original `options`.
    `rng` is a seed or a numpy Generator; reuse one Generator across calls.
    """
    perm = np.random.default_rng(rng).permutation(len(options))
    pick = judge([options[i] for i in perm])
    return None if pick is None else int(perm[pick])


def majority_vote(votes, tie=None):
    """Most common vote, ignoring None; `tie` if the top two are level or no votes."""
    top = Counter(v for v in votes if v is not None).most_common(2)
    if not top or (len(top) == 2 and top[0][1] == top[1][1]):
        return tie
    return top[0][0]
