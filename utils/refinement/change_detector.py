"""Assess whether a corrected prediction should be accepted."""

from __future__ import annotations


class ChangeDetector:
    """Apply heuristic rules to decide on accepting corrections."""

    def __init__(self, min_delta: float = 0.15, min_confidence: float = 0.80) -> None:
        self.min_delta = min_delta
        self.min_confidence = min_confidence

    def evaluate(
        self,
        original_label: str,
        original_confidence: float,
        corrected_label: str,
        corrected_confidence: float,
    ) -> tuple[bool, str]:
        """Return acceptance flag and textual reason."""

        if corrected_label == original_label:
            return False, "label unchanged"
        if corrected_confidence - original_confidence < self.min_delta:
            return False, "confidence delta too low"
        if corrected_confidence < self.min_confidence:
            return False, "corrected confidence below threshold"
        return True, "label changed with sufficient confidence"
