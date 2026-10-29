"""Agreement between raters (judge vs. human, judge vs. judge) and judge helpers."""

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

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


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def krippendorff_alpha(ratings: ArrayLike, level: str = "nominal") -> float:
    """Krippendorff's alpha. `ratings` is (n_items, n_raters); None or NaN = missing.

    Unlike Cohen's/Fleiss' kappa it allows missing ratings and any number of
    raters, and `level` respects the scale: "nominal" (labels), "ordinal"
    (ranks: 1 < 2 < 3, distances follow how often each value is used) or
    "interval" (numeric scores, squared differences). Items with fewer than
    two ratings are ignored. Computed from the coincidence matrix
    (Krippendorff 2011).
    """
    r = np.asarray(ratings, dtype=object)
    if r.ndim != 2:
        raise ValueError("ratings must be (n_items, n_raters)")
    present = ~np.vectorize(_is_missing, otypes=[bool])(r)
    vals = r[present].tolist()
    if level == "nominal":
        cats: list[Any] = list(dict.fromkeys(vals))
    elif level in ("ordinal", "interval"):
        vals = [float(v) for v in vals]
        cats = sorted(set(vals))
    else:
        raise ValueError("level must be 'nominal', 'ordinal' or 'interval'")
    index = {c: i for i, c in enumerate(cats)}
    counts = np.zeros((len(r), len(cats)))
    for (u, _), v in zip(np.argwhere(present), vals, strict=True):
        counts[u, index[v]] += 1
    m = counts.sum(1)
    counts, m = counts[m >= 2], m[m >= 2]
    w = counts / (m - 1)[:, None]
    o = w.T @ counts - np.diag(w.sum(0))  # coincidences between pairable values
    nc = o.sum(1)
    n = nc.sum()
    if level == "nominal":
        delta = 1 - np.eye(len(cats))
    elif level == "interval":
        v = np.array(cats)
        delta = (v[:, None] - v[None, :]) ** 2
    else:
        cum = np.cumsum(nc)
        lo, hi = (
            np.minimum.outer(range(len(cats)), range(len(cats))),
            np.maximum.outer(range(len(cats)), range(len(cats))),
        )
        delta = (cum[hi] - cum[lo] + (nc[lo] - nc[hi]) / 2) ** 2
    expected = (np.outer(nc, nc) * delta).sum()
    return float(1 - (n - 1) * (o * delta).sum() / expected) if expected > 0 else math.nan


def spearman(a: ArrayLike, b: ArrayLike) -> float:
    """Spearman rank correlation, e.g. judge scores vs. human scores on the same items."""
    return float(stats.spearmanr(a, b).statistic)


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


# --- correcting a metric for judge error -----------------------------------


@dataclass(frozen=True)
class JudgeAdjustedRate:
    """Result of `judge_adjusted_rate`. `print()` it for the summary."""

    raw: float
    raw_ci: tuple[float, float]
    adjusted: float
    adjusted_ci: tuple[float, float]
    adjusted_ci_known_error: tuple[float, float]
    sensitivity: float
    specificity: float
    n_eval: int
    n_calib: int
    alpha: float

    def __str__(self) -> str:
        conf = round(100 * (1 - self.alpha))

        def ci(c: tuple[float, float]) -> str:
            return f"{conf}% CI {c[0]:.1%} to {c[1]:.1%}"

        def width(c: tuple[float, float]) -> float:
            return (c[1] - c[0]) * 100

        w0, w1, w2 = map(width, (self.raw_ci, self.adjusted_ci_known_error, self.adjusted_ci))
        return "\n".join(
            [
                f"Judge pass rate: {self.raw:.1%} ({ci(self.raw_ci)}), {self.n_eval} items.",
                f"Judge vs. human on {self.n_calib} calibration items: "
                f"sensitivity {self.sensitivity:.2f}, specificity {self.specificity:.2f}.",
                f"Adjusted pass rate: {self.adjusted:.1%} ({ci(self.adjusted_ci)}).",
                f"CI width: {w0:.1f} points if the judge were perfect, {w1:.1f} after "
                f"correcting for its error rates, {w2:.1f} also counting the uncertainty "
                f"in those rates ({w2 / w0:.1f}x).",
            ]
        )


def _binary(x: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, float)
    if arr.ndim != 1 or not np.isin(arr, [0, 1]).all():
        raise ValueError(f"{name} must be 1-D 0/1 labels (1 = pass)")
    return arr


def _rogan_gladen(raw: Any, se: Any, sp: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        youden = se + sp - 1
        return np.where(youden > 0, np.clip((raw + sp - 1) / youden, 0, 1), np.nan)


def judge_adjusted_rate(
    judge_labels: ArrayLike,
    calib_judge: ArrayLike,
    calib_human: ArrayLike,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> JudgeAdjustedRate:
    """Pass rate corrected for the judge's errors, with honest uncertainty.

    `judge_labels`: the judge's 0/1 verdicts on the eval set. `calib_judge`,
    `calib_human`: judge and human verdicts on a separate calibration set.
    The judge's sensitivity and specificity measured there correct the raw pass
    rate (Rogan & Gladen 1978). The CI bootstraps both sets, so it includes
    the uncertainty in those error rates; the summary shows how much each
    source widens it. Answers: how much uncertainty does the judge add?

    Assumes the judge's error rates are the same on the eval set as on the
    calibration set, and do not depend on the variant being judged. If prompt
    B produces answers the judge misreads more often, the correction is wrong.
    Estimates are clipped to [0, 1].
    """
    j = _binary(judge_labels, "judge_labels")
    cj, ch = _binary(calib_judge, "calib_judge"), _binary(calib_human, "calib_human")
    if len(cj) != len(ch):
        raise ValueError("calib_judge and calib_human must label the same items")
    if ch.min() == ch.max():
        raise ValueError("the calibration set needs both human passes and human fails")
    se, sp = float(cj[ch == 1].mean()), float(1 - cj[ch == 0].mean())
    if se + sp <= 1:
        raise ValueError("the judge is no better than chance on the calibration set")
    rng = np.random.default_rng(seed)
    n, m = len(j), len(cj)
    raw_b = j[rng.integers(0, n, (n_boot, n))].mean(1)
    idx = rng.integers(0, m, (n_boot, m))
    cjb, pos = cj[idx], ch[idx] == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        se_b = (cjb * pos).sum(1) / pos.sum(1)
        sp_b = ((1 - cjb) * ~pos).sum(1) / (~pos).sum(1)
    q = [50 * alpha, 100 - 50 * alpha]

    def ci(boots: Any) -> tuple[float, float]:
        lo, hi = np.nanpercentile(boots, q)
        return float(lo), float(hi)

    raw = float(j.mean())
    return JudgeAdjustedRate(
        raw=raw,
        raw_ci=ci(raw_b),
        adjusted=float(_rogan_gladen(raw, se, sp)),
        adjusted_ci=ci(_rogan_gladen(raw_b, se_b, sp_b)),
        adjusted_ci_known_error=ci(_rogan_gladen(raw_b, se, sp)),
        sensitivity=se,
        specificity=sp,
        n_eval=n,
        n_calib=m,
        alpha=alpha,
    )
