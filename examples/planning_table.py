"""Items needed to detect a pass-rate difference, and what a given N can detect.

Run: uv run python examples/planning_table.py
"""

from llm_judge_stats import n_for_proportions, smallest_detectable_difference

BASE, DISC = 0.70, 0.20

print(f"base pass rate {BASE:.0%}, variants disagree on {DISC:.0%} of items, alpha = 0.05\n")
print(
    "| Difference | paired, power 0.80 | paired, 0.90 | paired, 0.95 "
    "| unpaired, 0.80 (per variant) |"
)
print("|---|---|---|---|---|")
for d in (0.02, 0.05, 0.10, 0.15):
    paired = [
        n_for_proportions(BASE, BASE + d, power=p, discordance=DISC) for p in (0.8, 0.9, 0.95)
    ]
    unpaired = n_for_proportions(BASE, BASE + d, power=0.8)
    print(f"| {d * 100:.0f} points | " + " | ".join(map(str, paired)) + f" | {unpaired} |")

ns = [50, 100, 200, 500, 1000, 2000]
curve = smallest_detectable_difference(ns, p_base=BASE, discordance=DISC)
print("\nsmallest detectable difference, paired, power 0.80:")
for n, d in zip(ns, curve, strict=True):
    print(f"  n = {n:>4}: {d * 100:.1f} points")
