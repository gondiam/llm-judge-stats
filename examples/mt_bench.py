"""GPT-4-as-judge vs. human experts on MT-Bench (public dataset, CC-BY-4.0).

Data: https://huggingface.co/datasets/lmsys/mt_bench_human_judgments
Run:  pip install "llm-judge-stats[example]" && python examples/mt_bench.py
"""

import pandas as pd

from llm_judge_stats import (
    bootstrap_ci,
    cohen_kappa,
    f1,
    fleiss_kappa,
    majority_vote,
    mcc,
    n_for_kappa,
    n_for_precision,
)

URL = (
    "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/"
    "refs%2Fconvert%2Fparquet/default/{}/0000.parquet"
)
KEY = ["question_id", "model_a", "model_b", "turn"]
cols = KEY + ["winner", "judge"]
human = pd.read_parquet(URL.format("human"), columns=cols)
gpt4 = pd.read_parquet(URL.format("gpt4_pair"), columns=cols)

# GPT-4 was run in both orders; "tie (inconsistent)" = verdict flipped when swapped.
print(f"GPT-4 position-inconsistent verdicts: {(gpt4.winner == 'tie (inconsistent)').mean():.1%}")
gpt4["winner"] = gpt4.winner.replace({"tie (inconsistent)": "tie"})

# Several humans per item -> majority vote (ties between votes -> "tie").
gold = human.groupby(KEY).winner.agg(lambda v: majority_vote(v, tie="tie")).rename("human")
df = gpt4.set_index(KEY).join(gold, how="inner")
print(f"items judged by both: {len(df)}")

est, lo, hi = bootstrap_ci(cohen_kappa, df.winner, df.human, seed=0)
print(f"Cohen's kappa GPT-4 vs human majority: {est:.3f} [95% CI {lo:.3f}, {hi:.3f}]")

# Human-human agreement: items with >= 2 humans, first two votes.
pairs = human.groupby(KEY).winner.agg(list)
two = [v[:2] for v in pairs if len(v) >= 2]
est, lo, hi = bootstrap_ci(fleiss_kappa, two, seed=0)
print(f"Fleiss' kappa human-human ({len(two)} items): {est:.3f} [{lo:.3f}, {hi:.3f}]")

# Binary view on decisive items: "did model_a win?". Stratify by gold label.
dec = df[(df.human != "tie") & (df.winner != "tie")]
y, p = (dec.human == "model_a").astype(int), (dec.winner == "model_a").astype(int)
for name, fn in [("F1", f1), ("MCC", mcc)]:
    est, lo, hi = bootstrap_ci(fn, y, p, strata=y, seed=0)
    print(f"{name} (model_a wins, {len(dec)} decisive items): {est:.3f} [{lo:.3f}, {hi:.3f}]")

# How many items to label for +-0.05 on kappa?
print(
    "items for kappa +-0.05, from this pilot:",
    n_for_precision(cohen_kappa, df.winner, df.human, half_width=0.05, seed=0),
)
print(
    "items for binary kappa=0.6, prevalence=0.3, +-0.05 (planning formula):",
    n_for_kappa(0.6, 0.3, 0.05),
)
