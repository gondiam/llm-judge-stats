# llm-judge-stats

Statistics for turning LLM eval results into conclusions you can defend, or into
a clear statement that the data cannot support one. Small API, NumPy + SciPy, no
LLM frameworks.

The usual practice: compare two prompts on 40 examples, see a few points of
difference, declare a winner. No confidence interval, no power calculation, no
correction for having tried six prompts. This package answers four questions:

1. **Before running:** how many items do I need to detect the difference I care about?
2. **After running:** is the difference I see distinguishable from noise?
3. **About the judge:** how much does the judge agree with humans, and how much
   uncertainty does its disagreement add to my metric?
4. **For the report:** what can I honestly say in one paragraph?

```
pip install git+https://github.com/gondiam/llm-judge-stats
```

## The example that hurts

Two prompt variants with the **same** true pass rate, run on the same 40 items
(`examples/small_n.py`; the seed is the first one where B leads by 3 items):

```
A passes 28/40 (70.0%), B passes 31/40 (77.5%)
gap: +7.5 points
```

It looks like B wins. The paired analysis says:

```python
from llm_judge_stats import compare

print(compare(passed_a, passed_b, min_relevant=0.05))
```

```
B vs A: +7.5 points (70.0% -> 77.5%), 95% CI -7.5 to +21.7, 40 paired items.
Mid-p McNemar test p = 0.344; Cohen's h = 0.17.
Not distinguishable from noise at alpha = 0.05.
To detect a 5-point difference with 80% power you would need about 705 paired items.
```

In 10,000 runs of the same setup, with no real difference between the variants,
49.4% show a gap of 7.5 points or more in one direction or the other.

## How many items do you need?

Base pass rate 70%, the two variants disagree on 20% of items, two-sided
alpha = 0.05 (`examples/planning_table.py`). Paired = both variants run on the
same items, which is the normal case and roughly halves the items needed.

| Difference to detect | paired, power 0.80 | paired, 0.90 | paired, 0.95 | unpaired, 0.80 (per variant) |
|---|---|---|---|---|
| 2 points  | 3923 | 5250 | 6492 | 8080 |
| 5 points  | 626  | 837  | 1034 | 1251 |
| 10 points | 155  | 206  | 254  | 294  |
| 15 points | 68   | 90   | 110  | 121  |

The discordance (share of items where exactly one variant passes) drives the
paired numbers more than the pass rates do. Measure it on a pilot. The table
holds it at 20% for every row.

The reverse question, given the items you already have:

```python
from llm_judge_stats import smallest_detectable_difference

smallest_detectable_difference([50, 100, 200, 500, 1000, 2000], p_base=0.70, discordance=0.2)
```

| Items | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| Smallest detectable difference (power 0.80) | 17.3 pts | 12.4 | 8.8 | 5.6 | 4.0 | 2.8 |

## API

### 1. Planning (run before the eval)

| Function | What it does |
|---|---|
| `n_for_proportions(p_a, p_b, power=0.8, alpha=0.05, discordance=None)` | Items needed to detect a pass-rate difference. Paired (McNemar, Connor 1987) when `discordance` is given, unpaired (items per variant) otherwise |
| `n_for_means(delta, sd, power=0.8, alpha=0.05, paired=True)` | Same for continuous judge scores, exact via the noncentral t. For paired data `sd` is the SD of per-item differences |
| `smallest_detectable_difference(n, ...)` | The inverse: the smallest difference a given N detects, in either direction. Accepts an array of N for a detectability curve |
| `n_for_kappa(kappa, prevalence, half_width)` | Items to label so a binary kappa's CI has a target width (Bloch & Kraemer 1989) |
| `n_for_precision(stat, *pilot, half_width=...)` | Same for any statistic (F1, MCC, ICC, ...), extrapolated from a pilot |

### 2. Analysing an eval

| Function | What it does |
|---|---|
| `compare(a, b, paired=True, alpha=0.05, min_relevant=None)` | Difference, CI, p-value and effect size for two variants. Paired: mid-p McNemar + Agresti-Min CI for 0/1 data, paired t otherwise. Unpaired: two-proportion z + Agresti-Caffo CI, or Welch t. Returns a `Comparison`; `print()` gives the verdict |
| `compare_many(scores, baseline, correction="holm")` | k variants against a baseline, p-values adjusted by Holm (FWER) or Benjamini-Hochberg (FDR) |
| `adjust_pvalues(p, method="holm")` | Holm or Benjamini-Hochberg adjustment on its own |
| `bootstrap_ci(stat, *data, method="percentile", strata=None)` | Bootstrap CI over items for any metric; `method="bca"` for skewed statistics |

Effect size is always reported next to the p-value: Cohen's d_z (paired) or d
(unpaired) for scores, Cohen's h for pass rates.

When the difference is not distinguishable, the verdict says how many items
would detect `min_relevant` (the smallest difference that matters to you). Without
it, the verdict gives the smallest difference the current N can detect.
Planning on the observed difference ("post-hoc power") is circular, so the
package does not do it.

```python
from llm_judge_stats import compare_many

results = compare_many({"base": base, "v1": v1, "v2": v2, "v3": v3}, baseline="base")
print(*results, sep="\n\n")
```

### 3. Judge quality

| Function | What it does |
|---|---|
| `cohen_kappa(a, b)` | Agreement between two raters (any categories) |
| `fleiss_kappa(ratings)` | Agreement among m raters, `ratings` is (items, raters) |
| `krippendorff_alpha(ratings, level="nominal")` | Agreement with missing ratings allowed (None/NaN); `level` nominal, ordinal or interval |
| `icc(scores, kind="2,1")` | Shrout & Fleiss ICC for numeric scores |
| `spearman(a, b)` | Rank correlation between judge and human scores |
| `f1`, `mcc` | Binary F1 / Matthews correlation vs. gold |
| `judge_adjusted_rate(judge_labels, calib_judge, calib_human)` | Pass rate corrected for the judge's sensitivity and specificity, measured on a human-labelled calibration set (Rogan & Gladen 1978). The CI bootstraps both sets, and the summary shows how much each source widens it |
| `shuffled_judge(judge, options, rng)` | Shows options in random order, maps the pick back (counters position bias) |
| `majority_vote(votes, tie=None)` | Aggregates repeated judge samples or several annotators |

Any agreement statistic gets a CI through `bootstrap_ci`:

```python
bootstrap_ci(cohen_kappa, judge_labels, human_labels)  # (est, lo, hi)
bootstrap_ci(f1, gold, pred, strata=gold)  # keeps class balance fixed
```

### 4. Report

`print()` on a `compare` or `compare_many` result gives the verdict: estimated
difference, CI, test, effect size, whether it is distinguishable from noise, and
what more data would buy. `print()` on a `judge_adjusted_rate` result gives the
corrected rate and where its uncertainty comes from (example below).

## Example: GPT-4 vs. human experts on MT-Bench

```
uv run --extra example python examples/mt_bench.py
```

Uses the public [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) dataset (CC-BY-4.0):

```
GPT-4 position-inconsistent verdicts: 15.8%
items judged by both: 1232
Cohen's kappa GPT-4 vs human majority: 0.410 [95% CI 0.366, 0.454]
Fleiss' kappa human-human (761 items): 0.497 [0.446, 0.548]
F1 (model_a wins, 757 decisive items): 0.529 [0.444, 0.609]
MCC (model_a wins, 757 decisive items): 0.491 [0.404, 0.577]
items for kappa +-0.05, from this pilot: 936
items for binary kappa=0.6, prevalence=0.3, +-0.05 (planning formula): 1181
Krippendorff's alpha human-human, all votes (761 items, up to 5 raters): 0.478 [0.437, 0.523]
```

Correcting a judge-measured rate. On the 757 decisive items, a random half
plays the human-labelled calibration set, and on the other half only GPT-4's
verdict is used. The human answer for that half is known, so we can check
whether the correction lands on it:

```
Judge pass rate: 10.0% (95% CI 7.0% to 13.0%), 400 items.
Judge vs. human on 357 calibration items: sensitivity 0.46, specificity 0.98.
Adjusted pass rate: 18.3% (95% CI 10.4% to 29.8%).
CI width: 6.0 points if the judge were perfect, 13.7 after correcting for its error rates, 19.3 also counting the uncertainty in those rates (3.2x).
human rate on the GPT-4-only half (the target): 19.0%
```

GPT-4 misses more than half of the human "model_a wins" verdicts, so its raw
rate (10.0%) is far from the human one (19.0%), and its 95% CI excludes it.
The corrected rate covers the human value, but the honest interval is 3.2 times
wider than the naive one. That is what this judge costs. This is one split of
one dataset. The simulations below check that the CI coverage holds in general.

## Does it do what it promises?

`examples/simulations.py` (about 35 s) checks each method on simulated data
where the truth is known. Each rate comes with its Monte Carlo SE. The test
suite checks the same properties with fewer runs.

**`compare()` on null data** (no real difference), alpha = 0.05, 4000 runs each:

| Setting | Type I error | 95% CI coverage |
|---|---|---|
| paired pass rates, n = 40, p = 0.7 | 0.036 (SE 0.003) | 0.964 (SE 0.003) |
| unpaired pass rates, n = 40 + 40, p = 0.7 | 0.045 (SE 0.003) | 0.956 (SE 0.003) |
| paired scores, n = 30 | 0.053 (SE 0.004) | 0.947 (SE 0.004) |
| unpaired scores, n = 30 + 30, SDs 1 and 3 | 0.054 (SE 0.004) | 0.946 (SE 0.004) |

The paired pass-rate test stays below alpha at n = 40 because the discrete
mid-p test is mildly conservative at small n.

**Planned power is reached.** At the n from `n_for_proportions` (base 70%,
discordance 20%, target 0.80), `compare()` rejects in 0.793 (SE 0.006) of runs
for a 5-point difference (n = 626) and 0.797 (SE 0.006) for 10 points (n = 155).
An earlier version used the exact McNemar test. It reached only about 0.75
power and a 0.018 type I error, so it was replaced by mid-p.

**Bootstrap on a skewed statistic** (mean of 20 exponential draws, 1000 runs):
percentile coverage 0.892 (SE 0.010), BCa 0.912 (SE 0.009). BCa helps but
neither reaches 0.95 at n = 20. With small skewed samples, treat any
bootstrap CI as too narrow.

**`judge_adjusted_rate()`** (true rate 0.60, judge sensitivity 0.80,
specificity 0.95, 500 eval and 300 calibration items, 1000 runs): the raw judge
CI covers the true rate in 0.009 (SE 0.003) of runs, and the adjusted CI in 0.948
(SE 0.007).

## Assumptions

Every function's docstring states its assumptions and what happens when they
fail. The ones that matter most:

- **Items are independent.** Several turns of one conversation, or paraphrases
  of one question, are not independent items. CIs will be too narrow.
- **Paired tests need the same items in the same order** for both variants.
- **Planning formulas for pass rates are normal approximations**, and paired
  planning needs a discordance rate you can only estimate. `n_for_means` is exact
  under normality.
- **`judge_adjusted_rate` assumes the judge's errors do not depend on the
  variant** (non-differential misclassification), and that the calibration set
  looks like the eval set. If prompt B produces answers the judge misreads more
  often, the correction is wrong.

## Limitations

What this package does not do:

- **Dependence between items.** No cluster bootstrap or mixed models. If your
  items are grouped (conversations, documents, templates), the CIs are optimistic.
- **Provider model drift.** A result holds for the model version and date you
  ran. Nothing here detects that the API model changed underneath you.
- **Judge biases.** Position bias has one mitigation (`shuffled_judge`).
  Verbosity, self-preference and language bias are not measured here.
- **Multiple comparisons you did not declare.** `compare_many` corrects for the
  variants you pass it. It cannot correct for the twenty you tried and dropped.
  Its CIs are not adjusted.
- **Judge-error correction is binary only.** Continuous judge scores have no
  reliability correction yet.
- **It does not tell you which metric to use**, only how uncertain it is.

## Development

```
uv sync
uv run pytest            # 35 tests, ~10 s
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

The tests check against published values (Fleiss' 1971 example via Wikipedia,
Shrout & Fleiss 1979 ICCs, Krippendorff 2011 worked example at three levels,
G*Power sample sizes, SciPy's BCa). They also check by simulation the type I
error, CI coverage and power described above.

MIT licensed.
