"""Utilities for combining multiple ranking scores."""

from __future__ import annotations

from typing import Dict


class ScoreCombiner:
    """Combine scores using a weighted sum strategy.

    Parameters
    ----------
    weights:
        Mapping of score name to weight. The default gives full weight to the
        initial similarity score and ignores other scores.
    """

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = weights or {"similarity": 1.0}

    def combine(self, scores: Dict[str, float]) -> float:
        """Return a weighted sum of ``scores``.

        Missing score keys are treated as ``0.0``.
        """

        total = 0.0
        for name, weight in self.weights.items():
            total += scores.get(name, 0.0) * weight
        return total
