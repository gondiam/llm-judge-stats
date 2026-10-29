"""Agreement between raters (judge vs. human, judge vs. judge) and judge helpers."""

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike

T = TypeVar("T")


def cohen_kappa(a: ArrayLike, b: ArrayLike) -> float:
    """Cohen's kappa between two raters' labels (any hashable categories)."""
    x, y = np.asarray(a), np.asarray(b)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("a and b must be 1-D and the same length")
    cats = np.union1d(x, y)
    pe = ((x[:, None] == cats).mean(0) * (y[:, None] == cats).mean(0)).sum()
    return float((np.mean(x == y) - pe) / (1 - pe)) if pe < 1 else math.nan


def fleiss_kappa(ratings: ArrayLike) -> float:
    """Fleiss' kappa. `ratings` is (n_items, n_raters) of labels, no missing values."""
    r = np.asarray(ratings)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("ratings must be (n_items, n_raters) with n_raters >= 2")
    n, m = r.shape
    counts = (r[:, :, None] == np.unique(r)).sum(1)  # (n_items, n_categories)
    p_item = ((counts**2).sum(1) - m) / (m * (m - 1))
    pe = ((counts.sum(0) / (n * m)) ** 2).sum()
    return float((p_item.mean() - pe) / (1 - pe)) if pe < 1 else math.nan


def icc(scores: ArrayLike, kind: str = "2,1") -> float:
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


def _confusion(y_true: ArrayLike, y_pred: ArrayLike, positive: Any) -> list[float]:
    t, p = np.asarray(y_true) == positive, np.asarray(y_pred) == positive
    return [float(v) for v in ((t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum())]


def f1(y_true: ArrayLike, y_pred: ArrayLike, positive: Any = 1) -> float:
    """Binary F1 for `positive`; NaN when there are no positives at all."""
    tp, fp, fn, _ = _confusion(y_true, y_pred, positive)
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else math.nan


def mcc(y_true: ArrayLike, y_pred: ArrayLike, positive: Any = 1) -> float:
    """Binary Matthews correlation; 0 when a marginal is empty (sklearn convention)."""
    tp, fp, fn, tn = _confusion(y_true, y_pred, positive)
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


# --- running judges -------------------------------------------------------


def shuffled_judge(
    judge: Callable[[list[T]], int | None],
    options: Sequence[T],
    rng: int | np.random.Generator | None = None,
) -> int | None:
    """Show `options` to `judge` in random order to neutralise position bias.

    `judge(list) -> index into that list` (or None for a tie). The pick is
    mapped back to an index into the original `options`.
    `rng` is a seed or a numpy Generator; reuse one Generator across calls.
    """
    perm = np.random.default_rng(rng).permutation(len(options))
    pick = judge([options[i] for i in perm])
    return None if pick is None else int(perm[pick])


def majority_vote(votes: Iterable[T | None], tie: T | None = None) -> T | None:
    """Most common vote, ignoring None; `tie` if the top two are level or no votes."""
    top = Counter(v for v in votes if v is not None).most_common(2)
    if not top or (len(top) == 2 and top[0][1] == top[1][1]):
        return tie
    return top[0][0]
