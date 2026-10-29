"""Statistical tools for LLM evals: planning, comparing variants and checking judges."""

from .agreement import cohen_kappa, f1, fleiss_kappa, icc, majority_vote, mcc, shuffled_judge
from .planning import (
    n_for_kappa,
    n_for_means,
    n_for_precision,
    n_for_proportions,
    smallest_detectable_difference,
)
from .resampling import bootstrap_ci

__all__ = [
    "cohen_kappa",
    "fleiss_kappa",
    "icc",
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
__version__ = "0.1.0"
