"""Compute confidence scores from probability distributions."""
from __future__ import annotations

from typing import Dict


class ConfidenceScorer:
    """Derive a confidence measure for a predicted label."""

    def compute(self, probs: Dict[str, float], label: str) -> float:
        """Return the probability associated with ``label``."""
        return float(probs.get(label, 0.0))
