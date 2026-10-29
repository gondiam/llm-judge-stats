"""Do the methods do what they promise? Coverage, type I error and power by simulation.

Run: uv run python examples/simulations.py   (a few minutes)
Every rate is printed with its Monte Carlo standard error.
"""

import math

import numpy as np

from llm_judge_stats import bootstrap_ci, compare, judge_adjusted_rate, n_for_proportions

rng = np.random.default_rng(0)


def rate(hits: list[bool]) -> str:
    p = float(np.mean(hits))
    return f"{p:.3f} (MC SE {math.sqrt(p * (1 - p) / len(hits)):.3f}, {len(hits)} runs)"


def paired_binary(n: int, p: float, disc: float, d: float) -> tuple[np.ndarray, np.ndarray]:
    cells = [(disc + d) / 2, (disc - d) / 2, p - (disc - d) / 2]  # b-only, a-only, both
    c = rng.choice(4, size=n, p=cells + [1 - sum(cells)])
    return (c == 1) | (c == 2), (c == 0) | (c == 2)


print("compare(): null data, alpha = 0.05 - type I error and 95% CI coverage")
settings = {
    "paired pass rates, n=40, p=0.7": (lambda: paired_binary(40, 0.7, 0.2, 0.0), True),
    "unpaired pass rates, n=40+40, p=0.7": (
        lambda: (rng.binomial(1, 0.7, 40), rng.binomial(1, 0.7, 40)),
        False,
    ),
    "paired scores, n=30": (
        lambda: (lambda x: (x, x + rng.normal(0, 1, 30)))(rng.normal(5, 2, 30)),
        True,
    ),
    "unpaired scores, n=30+30, SDs 1 and 3": (
        lambda: (rng.normal(5, 1, 30), rng.normal(5, 3, 30)),
        False,
    ),
}
for name, (draw, paired) in settings.items():
    res = [compare(*draw(), paired=paired) for _ in range(4000)]
    print(f"  {name}")
    print(f"    type I error: {rate([r.p_value < 0.05 for r in res])}")
    print(f"    CI coverage:  {rate([r.ci[0] <= 0 <= r.ci[1] for r in res])}")

print("\nn_for_proportions(): power of compare() (mid-p McNemar) at the planned n (target 0.80)")
for d in (0.05, 0.10):
    n = n_for_proportions(0.7, 0.7 + d, discordance=0.2)
    hits = [compare(*paired_binary(n, 0.7, 0.2, d)).p_value < 0.05 for _ in range(4000)]
    print(f"  {d * 100:.0f} points, n = {n}: {rate(hits)}")

print("\nbootstrap_ci(): 95% coverage of the mean of a skewed (exponential) sample, n = 20")
pct, bca = [], []
for _ in range(1000):
    x = rng.exponential(size=20)
    pct.append(bootstrap_ci(np.mean, x, n_boot=1000, seed=rng)[1:])
    bca.append(bootstrap_ci(np.mean, x, n_boot=1000, method="bca", seed=rng)[1:])
print(f"  percentile: {rate([lo <= 1 <= hi for lo, hi in pct])}")
print(f"  BCa:        {rate([lo <= 1 <= hi for lo, hi in bca])}")

print("\njudge_adjusted_rate(): true pass rate 0.60, judge sensitivity 0.80, specificity 0.95")
print("  500 eval items, 300 calibration items")
raw, adj = [], []
for _ in range(1000):
    truth, calib = rng.random(500) < 0.6, rng.random(300) < 0.6

    def judge(t: np.ndarray) -> np.ndarray:
        return np.where(t, rng.random(len(t)) < 0.8, rng.random(len(t)) > 0.95)

    r = judge_adjusted_rate(judge(truth), judge(calib), calib, n_boot=1000, seed=rng)
    raw.append(r.raw_ci[0] <= 0.6 <= r.raw_ci[1])
    adj.append(r.adjusted_ci[0] <= 0.6 <= r.adjusted_ci[1])
print(f"  coverage of the raw judge CI:      {rate(raw)}")
print(f"  coverage of the adjusted CI:       {rate(adj)}")
print("  last run:")
print("    " + str(r).replace("\n", "\n    "))
