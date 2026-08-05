"""Small statistics shared by metric-selection implementations."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(scores, labels):
    """Tie-aware binary AUROC with pairwise deletion of non-finite scores."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(scores, method="average")
    numerator = ranks[labels == 1].sum() - positives * (positives + 1) / 2
    return float(numerator / (positives * negatives))
