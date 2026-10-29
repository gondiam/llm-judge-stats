"""Statistical tools for LLM evals: planning, comparing variants and checking judges."""

from .agreement import (
    JudgeAdjustedRate,
    cohen_kappa,
    f1,
    fleiss_kappa,
    icc,
    judge_adjusted_rate,
    krippendorff_alpha,
    majority_vote,
    mcc,
    shuffled_judge,
    spearman,
)
from .compare import Comparison, adjust_pvalues, compare, compare_many
from .planning import (
    n_for_kappa,
    n_for_means,
    n_for_precision,
    n_for_proportions,
    smallest_detectable_difference,
)
from .resampling import bootstrap_ci

__all__ = [
    "compare",
    "compare_many",
    "Comparison",
    "adjust_pvalues",
    "cohen_kappa",
    "fleiss_kappa",
    "icc",
    "krippendorff_alpha",
    "spearman",
    "judge_adjusted_rate",
    "JudgeAdjustedRate",
    "f1",
    "mcc",
    "bootstrap_ci",
    "n_for_kappa",
    "n_for_precision",
    "n_for_proportions",
    "n_for_means",
    "smallest_detectable_difference",
    "shuffled_judge",
    "majority_vote",
]
__version__ = "0.2.0"
