"""Decode raw logits into per-label probabilities."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple


class LogitDecoder:
    """Convert final-token scores into a label probability distribution."""

    def __init__(self, labels: Iterable[str] | None = None) -> None:
        self.labels = list(labels or ["primary", "secondary", "none"])

    def decode(self, scores: Sequence[Sequence[float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return ``(probabilities, logits)`` for each label.

        ``scores`` is expected to be a sequence where the last element
        contains a vector of scores for the vocabulary.
        """

        if not scores:
            raise ValueError("No scores provided")
        last_scores = scores[-1]
        logits: Dict[str, float] = {}
        for lbl in self.labels:
            idx = self._token_id(lbl)
            logits[lbl] = float(last_scores[idx])
        prob_values = self._softmax(list(logits.values()))
        probs = {lbl: prob for lbl, prob in zip(self.labels, prob_values)}
        return probs, logits

    def _token_id(self, label: str) -> int:
        """Deterministic pseudo token id for a label string."""
        return abs(hash(label)) % 10000

    @staticmethod
    def _softmax(values: List[float]) -> List[float]:
        max_val = max(values)
        exp_vals = [math.exp(v - max_val) for v in values]
        total = sum(exp_vals)
        return [v / total for v in exp_vals]
