from __future__ import annotations

"""Compute robustness scores for perturbation tests."""

from dataclasses import dataclass
from typing import List

from .llama3_inference import LLMResult


@dataclass
class ScoreResult:
    """Calculated metrics for a perturbation run."""

    invariance_score: float
    avg_confidence_drop: float


class ScoreComputer:
    """Calculate invariance and confidence-drop metrics."""

    def compute(
        self,
        original_confidence: float,
        original_label: str,
        results: List[LLMResult],
    ) -> ScoreResult:
        if not results:
            return ScoreResult(0.0, 0.0)
        match_count = sum(
            1 for r in results if r.predicted_label == original_label
        )
        invariance = match_count / len(results)
        avg_drop = (
            sum(original_confidence - r.confidence for r in results)
            / len(results)
        )
        return ScoreResult(
            invariance_score=invariance,
            avg_confidence_drop=avg_drop,
        )
