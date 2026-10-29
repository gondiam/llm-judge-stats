"""Two prompt variants, 40 items, no real difference - but one seems to win.

Each item has its own difficulty, shared by both variants (that is what makes
the comparison paired). Both variants have the same true pass rate. SEED is
the first seed where B happens to lead by 3 items (7.5 points); the last
block shows how often that happens by chance.
Run: uv run python examples/small_n.py
"""

import numpy as np

from llm_judge_stats import compare

N, SEED = 40, 3


def run(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    difficulty = rng.beta(3, 1.3, N)  # per-item pass probability, mean ~0.70
    return rng.random(N) < difficulty, rng.random(N) < difficulty


a, b = run(np.random.default_rng(SEED))
gap = b.mean() - a.mean()
print(f"A passes {a.sum()}/{N} ({a.mean():.1%}), B passes {b.sum()}/{N} ({b.mean():.1%})")
print(f"gap: {gap * 100:+.1f} points\n")
print(compare(a, b, min_relevant=0.05))

rng = np.random.default_rng(0)
gaps = np.array([np.subtract(*(x.mean() for x in run(rng))) for _ in range(10_000)])
print(
    f"\nIn 10,000 runs of the same null setup, {np.mean(np.abs(gaps) >= abs(gap) - 1e-9):.1%} "
    f"show a gap of {abs(gap) * 100:.1f} points or more in one direction or the other."
)
