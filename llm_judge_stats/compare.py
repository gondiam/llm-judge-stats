"""Is the difference between two eval variants distinguishable from noise?"""

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .planning import (
    _check_paired,
    n_for_means,
    n_for_proportions,
    smallest_detectable_difference,
)
from .resampling import _z


@dataclass(frozen=True)
class Comparison:
    """Result of `compare`. `print()` it for the verdict; fields for the numbers.

    `difference` is always b - a. `spread` is what sample-size planning needs:
    the discordance rate (paired pass rates), SD of per-item differences
    (paired scores) or pooled SD (unpaired scores); NaN for unpaired pass rates.
    """

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    difference: float
    ci: tuple[float, float]
    p_value: float
    effect_size: float
    effect_name: str
    test: str
    n_a: int
    n_b: int
    paired: bool
    binary: bool
    alpha: float
    spread: float
    p_adjusted: float | None = None
    correction: str | None = None
    n_needed: int | None = None
    planned_on: float | None = None
    detectable: float | None = None

    @property
    def label(self) -> str:
        return f"{self.name_b} vs {self.name_a}"

    @property
    def significant(self) -> bool:
        p = self.p_value if self.p_adjusted is None else self.p_adjusted
        return p < self.alpha

    def __str__(self) -> str:

        def f(x: float) -> str:
            return f"{x * 100:+.1f}" if self.binary else f"{x:+.2f}"

        unit = " points" if self.binary else ""
        means = (
            f"{self.mean_a:.1%} -> {self.mean_b:.1%}"
            if self.binary
            else f"mean {self.mean_a:.2f} -> {self.mean_b:.2f}"
        )
        items = f"{self.n_a} paired items" if self.paired else f"{self.n_a} vs {self.n_b} items"
        conf = round(100 * (1 - self.alpha))
        lines = [
            f"{self.label}: {f(self.difference)}{unit} ({means}), "
            f"{conf}% CI {f(self.ci[0])} to {f(self.ci[1])}, {items}.",
            f"{self.test} {_fmt_p(self.p_value)}; {self.effect_name} = {self.effect_size:.2f}.",
        ]
        if self.p_adjusted is not None:
            lines.append(f"{self.correction}-adjusted {_fmt_p(self.p_adjusted)} (CI not adjusted).")
        if self.significant:
            side = "higher" if self.difference > 0 else "lower"
            lines.append(
                f"Distinguishable from noise at alpha = {self.alpha}: {self.name_b} is {side}."
            )
        else:
            lines.append(f"Not distinguishable from noise at alpha = {self.alpha}.")
            per = "paired items" if self.paired else "items per variant"

            def size(d: float) -> str:
                return f"{d * 100:.3g}-point" if self.binary else f"{d:.2f}"

            def amount(d: float) -> str:
                return f"{d * 100:.3g} points" if self.binary else f"{d:.2f}"

            if self.n_needed is not None and self.planned_on is not None:
                lines.append(
                    f"To detect a {size(self.planned_on)} difference with 80% power "
                    f"you would need about {self.n_needed} {per}."
                )
            elif self.detectable is not None and math.isnan(self.detectable):
                lines.append(
                    f"With {self.n_a} {per} and the observed spread, no difference within "
                    "reach is detectable with 80% power."
                )
            elif self.detectable is not None:
                lines.append(
                    f"With {self.n_a} {per}, the smallest difference detectable with 80% "
                    f"power is about {amount(self.detectable)}. Pass min_relevant to get the "
                    "items needed for the difference you care about."
                )
        return "\n".join(lines)


def _fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _cohen_h(p_a: float, p_b: float) -> float:
    return 2 * math.asin(math.sqrt(p_b)) - 2 * math.asin(math.sqrt(p_a))


def _as_scores(x: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, float)
    if arr.ndim != 1 or len(arr) < 2 or np.isnan(arr).any():
        raise ValueError(f"{name} must be 1-D with at least 2 scores and no missing values")
    return arr


def _compare(
    a: ArrayLike,
    b: ArrayLike,
    paired: bool,
    alpha: float,
    binary: bool | None,
    names: tuple[str, str],
) -> Comparison:
    x, y = _as_scores(a, "a"), _as_scores(b, "b")
    if paired and len(x) != len(y):
        raise ValueError("paired comparison needs the same items, in the same order, in a and b")
    if binary is None:
        binary = bool(np.isin(np.concatenate([x, y]), [0, 1]).all())
    ma, mb, n, m = float(x.mean()), float(y.mean()), len(x), len(y)
    z = _z(alpha)
    common = dict(
        name_a=names[0], name_b=names[1], mean_a=ma, mean_b=mb, difference=mb - ma, n_a=n, n_b=m
    )

    if binary and paired:
        b_only, a_only = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
        disc = b_only + a_only
        p = stats.binomtest(b_only, disc, 0.5).pvalue if disc else 1.0
        # Agresti & Min (2005): add 1/2 to each cell of the 2x2 table, then Wald
        nn, p10, p01 = n + 2, (b_only + 0.5) / (n + 2), (a_only + 0.5) / (n + 2)
        d_adj = p10 - p01
        half = z * math.sqrt((p10 + p01 - d_adj**2) / nn)
        ci = (d_adj - half, d_adj + half)
        test, eff, ename, spread = "Exact McNemar test", _cohen_h(ma, mb), "Cohen's h", disc / n
    elif binary:
        pooled = (x.sum() + y.sum()) / (n + m)
        se0 = math.sqrt(pooled * (1 - pooled) * (1 / n + 1 / m))
        p = 2 * stats.norm.sf(abs(mb - ma) / se0) if se0 > 0 else 1.0
        # Agresti & Caffo (2000): add one pass and one fail to each variant, then Wald
        pa, pb = (x.sum() + 1) / (n + 2), (y.sum() + 1) / (m + 2)
        half = z * math.sqrt(pa * (1 - pa) / (n + 2) + pb * (1 - pb) / (m + 2))
        ci = (pb - pa - half, pb - pa + half)
        test, eff, ename, spread = "Two-proportion z test", _cohen_h(ma, mb), "Cohen's h", math.nan
    elif paired:
        d = y - x
        res = stats.ttest_1samp(d, 0)
        sd = float(d.std(ddof=1))
        ci_ = res.confidence_interval(1 - alpha)
        p, ci = float(res.pvalue) if sd > 0 else float(ma != mb), (ci_.low, ci_.high)
        test, ename, spread = "Paired t test", "Cohen's d_z", sd
        eff = (mb - ma) / sd if sd > 0 else math.nan
    else:
        res = stats.ttest_ind(y, x, equal_var=False)
        ci_ = res.confidence_interval(1 - alpha)
        sd = math.sqrt(((n - 1) * x.var(ddof=1) + (m - 1) * y.var(ddof=1)) / (n + m - 2))
        p, ci = float(res.pvalue), (ci_.low, ci_.high)
        test, ename, spread = "Welch t test", "Cohen's d", sd
        eff = (mb - ma) / sd if sd > 0 else math.nan

    return Comparison(
        **common,  # type: ignore[arg-type]
        ci=(float(ci[0]), float(ci[1])),
        p_value=float(p),
        effect_size=float(eff),
        effect_name=ename,
        test=test,
        paired=paired,
        binary=binary,
        alpha=alpha,
        spread=float(spread),
    )


def _with_plan(c: Comparison, min_relevant: float | None, alpha: float) -> Comparison:
    # Planning on the observed difference ("post-hoc power") is circular, so
    # without a target we report what this N can detect instead.
    disc = c.spread if c.binary and c.paired else None
    try:
        if min_relevant is None:
            if c.binary:
                if disc == 0:
                    return c
                d = smallest_detectable_difference(
                    c.n_a, p_base=c.mean_a, discordance=disc, alpha=alpha
                )
            elif c.spread > 0:
                d = smallest_detectable_difference(c.n_a, sd=c.spread, paired=c.paired, alpha=alpha)
            else:
                return c
            return dataclasses.replace(c, detectable=float(d))  # NaN: nothing in reach
        target, sign = abs(min_relevant), 1 if c.difference >= 0 else -1
        if c.binary:
            disc = None if disc is None else max(disc, target)
            p_b = c.mean_a + sign * target
            if disc is not None:
                _check_paired(c.mean_a, p_b, disc)
            n = n_for_proportions(c.mean_a, p_b, alpha=alpha, discordance=disc)
        elif c.spread > 0:
            n = n_for_means(target, c.spread, alpha=alpha, paired=c.paired)
        else:
            return c
    except ValueError:  # the target would push the pass rate outside [0, 1]
        return c
    return dataclasses.replace(c, n_needed=n, planned_on=target)


def compare(
    a: ArrayLike,
    b: ArrayLike,
    *,
    paired: bool = True,
    alpha: float = 0.05,
    binary: bool | None = None,
    min_relevant: float | None = None,
    names: tuple[str, str] = ("A", "B"),
) -> Comparison:
    """Compare variant b against variant a, one score per eval item.

    `names` labels them in the verdict.

    paired=True (the default, and the normal case): the same items, in the same
    order, scored under both variants. Pairing removes item difficulty from the
    noise and usually cuts the items needed several-fold.

    - 0/1 scores (pass/fail, auto-detected; override with `binary`): exact
      McNemar test, Agresti-Min CI for the difference in pass rates, Cohen's h.
      Unpaired: two-proportion z test, Agresti-Caffo CI.
    - other scores: paired t test on per-item differences, Cohen's d_z.
      Unpaired: Welch t test, Cohen's d. Assumes differences are roughly
      normal or n is large enough (~30+) for the mean to be.

    The verdict follows the p-value; the CI comes from a different method and
    can, in borderline cases, disagree by a hair. When b is not distinguishable,
    the result says how many items would be needed for 80% power to detect
    `min_relevant` - the smallest difference that would matter to you, in score
    units (0.05 = 5 points for pass rates). Without it, it says the smallest
    difference this N can detect. Planning reuses the observed discordance or
    SD, which are themselves estimates from this eval.
    """
    return _with_plan(_compare(a, b, paired, alpha, binary, names), min_relevant, alpha)


def adjust_pvalues(p: ArrayLike, method: str = "holm") -> NDArray[np.float64]:
    """Adjust p-values from k comparisons run on the same eval.

    "holm": controls the chance of *any* false winner (FWER); use it when each
    claimed winner will be acted on. "bh": Benjamini-Hochberg, controls the
    expected *share* of false winners (FDR); use it when screening many variants.
    BH assumes the tests are independent or positively correlated, which holds
    for variants compared against one shared baseline.
    """
    ps = np.asarray(p, float)
    if method == "bh":
        return np.asarray(stats.false_discovery_control(ps, method="bh"), float)
    if method != "holm":
        raise ValueError("method must be 'holm' or 'bh'")
    order = np.argsort(ps)
    steps = (len(ps) - np.arange(len(ps))) * ps[order]
    out = np.empty_like(ps)
    out[order] = np.minimum(1, np.maximum.accumulate(steps))
    return out


def compare_many(
    scores: Mapping[str, ArrayLike],
    baseline: str,
    *,
    correction: str = "holm",
    min_relevant: float | None = None,
    **kwargs: Any,
) -> list[Comparison]:
    """Compare every variant in `scores` against `baseline`, adjusting p-values.

    `kwargs` go to `compare`. Each result carries the adjusted p-value
    (`correction` as in `adjust_pvalues`) and its verdict uses it. CIs are not
    adjusted. Planning (items needed, smallest detectable difference) uses
    alpha / k (Bonferroni), which is conservative for both corrections.
    `print(*results, sep="\\n\\n")` for the report. It corrects only
    for the variants passed in, not for ones tried and dropped.
    """
    if baseline not in scores:
        raise ValueError(f"baseline {baseline!r} is not in scores")
    alpha = kwargs.pop("alpha", 0.05)
    paired, binary = kwargs.pop("paired", True), kwargs.pop("binary", None)
    if kwargs:
        raise TypeError(f"unexpected arguments: {', '.join(kwargs)}")
    base = scores[baseline]
    res = [
        _compare(base, v, paired, alpha, binary, (baseline, k))
        for k, v in scores.items()
        if k != baseline
    ]
    adj = adjust_pvalues([r.p_value for r in res], correction)
    name = {"holm": "Holm", "bh": "Benjamini-Hochberg"}[correction]
    return [
        _with_plan(
            dataclasses.replace(r, p_adjusted=float(pa), correction=name),
            min_relevant,
            alpha / len(res),
        )
        for r, pa in zip(res, adj, strict=True)
    ]
