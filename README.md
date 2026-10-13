# llm-judge-stats

Statistics for checking whether an LLM judge can be trusted. One file, NumPy only.

| Function | What it does |
|---|---|
| `cohen_kappa(a, b)` | Agreement between two raters (any categories) |
| `fleiss_kappa(ratings)` | Agreement among m raters, `ratings` is (items, raters) |
| `icc(scores, kind="2,1")` | Shrout & Fleiss ICC for numeric scores (`"1,1"`, `"2,1"`, `"3,1"`) |
| `f1`, `mcc` | Binary F1 / Matthews correlation vs. gold |
| `bootstrap_ci(stat, *data, strata=None)` | Percentile bootstrap 95% CI over items, optionally stratified |
| `n_for_kappa(kappa, prevalence, half_width)` | Items to label for a target CI width on binary kappa (planning formula, Bloch & Kraemer 1989) |
| `n_for_precision(stat, *pilot, half_width=...)` | Same thing for any statistic (F1, MCC, ICC, ...), extrapolated from a pilot |
| `shuffled_judge(judge, options, rng)` | Shows options in random order, maps the pick back (counters position bias) |
| `majority_vote(votes, tie=None)` | Aggregates repeated judge samples or several annotators |

```python
from llm_judge_stats import bootstrap_ci, cohen_kappa, f1, n_for_kappa

bootstrap_ci(cohen_kappa, judge_labels, human_labels)            # (est, lo, hi)
bootstrap_ci(f1, gold, pred, strata=gold)                        # keeps class balance fixed
n_for_kappa(kappa=0.6, prevalence=0.3, half_width=0.05)          # -> 1181 items
```

## Example: GPT-4 vs. human experts on MT-Bench

```
pip install -e ".[example]"
python examples/mt_bench.py
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
```

## Tests

```
pip install -e ".[test]" && pytest
```

The tests check against published values (Fleiss' 1971 example via Wikipedia, Shrout & Fleiss 1979 ICCs) and check the sample-size formula by simulation.

MIT licensed.
