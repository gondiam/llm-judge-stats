"""GPT-4-as-judge vs. human experts on MT-Bench (public dataset, CC-BY-4.0).

Data: https://huggingface.co/datasets/lmsys/mt_bench_human_judgments
Run:  uv run --extra example python examples/mt_bench.py
"""

import numpy as np
import pandas as pd

from llm_judge_stats import (
    bootstrap_ci,
    cohen_kappa,
    f1,
    fleiss_kappa,
    judge_adjusted_rate,
    krippendorff_alpha,
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
FLIP = {"model_a": "model_b", "model_b": "model_a"}


def reorder(df: pd.DataFrame, swap: pd.Series) -> pd.DataFrame:
    """Swap model_a and model_b (and the winner) on the rows where `swap` is true."""
    df = df.copy()
    df["winner"] = df.winner.where(~swap, df.winner.map(FLIP).fillna(df.winner))
    a, b = df.model_a.where(~swap, df.model_b), df.model_b.where(~swap, df.model_a)
    df["model_a"], df["model_b"] = a, b
    return df


human = pd.read_parquet(URL.format("human"), columns=cols)
gpt4 = pd.read_parquet(URL.format("gpt4_pair"), columns=cols)

# GPT-4 was run in both orders; "tie (inconsistent)" = verdict flipped when swapped.
print(f"GPT-4 position-inconsistent verdicts: {(gpt4.winner == 'tie (inconsistent)').mean():.1%}")
gpt4["winner"] = gpt4.winner.replace({"tie (inconsistent)": "tie"})

# Human rows list each model pair in either order, GPT-4 rows in one fixed order.
# Put every pair in alphabetical order so the same comparison gets the same key.
# Without this a third of the items judged by both do not match.
gpt4_order = set(zip(gpt4.model_a, gpt4.model_b, strict=True))  # pairs as GPT-4 saw them
human = reorder(human, human.model_a > human.model_b)
gpt4 = reorder(gpt4, gpt4.model_a > gpt4.model_b)

# Several humans per item -> majority vote (ties between votes -> "tie").
gold = human.groupby(KEY).winner.agg(lambda v: majority_vote(v, tie="tie")).rename("human")
df = gpt4.set_index(KEY).join(gold, how="inner")
print(f"items judged by both: {len(df)}")

print(f"raw agreement GPT-4 vs human majority: {(df.winner == df.human).mean():.1%}")
est, lo, hi = bootstrap_ci(cohen_kappa, df.winner, df.human, seed=0)
print(f"Cohen's kappa GPT-4 vs human majority: {est:.3f} [95% CI {lo:.3f}, {hi:.3f}]")

# Kappa depends on which model is called model_a. In the order GPT-4 saw the pairs,
# model_b is usually the stronger model, the labels are unbalanced and kappa drops,
# with the same data and the same raw agreement.
as_seen = df.reset_index()
pairs_ab = zip(as_seen.model_a, as_seen.model_b, strict=True)
swap = pd.Series([(b, a) in gpt4_order for a, b in pairs_ab])
as_seen["human"] = as_seen.human.where(~swap, as_seen.human.map(FLIP).fillna(as_seen.human))
as_seen = reorder(as_seen, swap)
est, lo, hi = bootstrap_ci(cohen_kappa, as_seen.winner, as_seen.human, seed=0)
print(f"  same items, pairs in the order GPT-4 saw them: {est:.3f} [{lo:.3f}, {hi:.3f}]")

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

# Krippendorff's alpha uses every human vote, with missing ones where an item had fewer.
votes = pairs[pairs.map(len) >= 2]
width = votes.map(len).max()
padded = [v + [None] * (width - len(v)) for v in votes]
est, lo, hi = bootstrap_ci(krippendorff_alpha, padded, n_boot=500, seed=0)
print(
    f"Krippendorff's alpha human-human, all votes ({len(padded)} items, up to {width} raters): "
    f"{est:.3f} [{lo:.3f}, {hi:.3f}]"
)

# How often does model_a win? Use the order GPT-4 saw the pairs, where model_a rarely
# wins and GPT-4 misses many of those wins. Pretend humans labelled only half of the
# decisive items: that half calibrates GPT-4, the other half is judged by GPT-4 alone.
# The human rate on the second half is known here, so we can see whether the correction lands.
dec = as_seen[(as_seen.human != "tie") & (as_seen.winner != "tie")]
y, p = (dec.human == "model_a").astype(int), (dec.winner == "model_a").astype(int)
half = np.random.default_rng(0).random(len(dec)) < 0.5
r = judge_adjusted_rate(p[~half], p[half], y[half], seed=0)
print(f"\nRate of 'model_a wins' on {len(dec)} decisive items, pairs as GPT-4 saw them:")
print(r)
print(f"human rate on the GPT-4-only half (the target): {y[~half].mean():.1%}")
