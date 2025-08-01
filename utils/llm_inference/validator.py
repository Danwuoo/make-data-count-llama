"""Validation utilities for LLaMA3 inference results."""
from __future__ import annotations

from typing import Dict, Iterable

LABELS = {"primary", "secondary", "none"}


class InferenceValidator:
    """Simple validator ensuring outputs are well-formed."""

    def validate(self, result: Dict[str, object]) -> None:
        label = result.get("predicted_label")
        if label not in LABELS:
            raise ValueError(f"Invalid label: {label}")


class LabelValidator:
    """Validate labels and confidence produced by the decoder."""

    def __init__(self, allowed: Iterable[str] | None = None, min_confidence: float = 0.0) -> None:
        self.allowed = set(allowed or LABELS)
        self.min_confidence = min_confidence

    def validate(self, label: str, confidence: float) -> None:
        """Raise ``ValueError`` if ``label`` or ``confidence`` are invalid."""
        if label not in self.allowed:
            raise ValueError(f"Invalid label: {label}")
        if confidence < self.min_confidence:
            raise ValueError(
                f"Confidence {confidence:.2f} below threshold {self.min_confidence:.2f}"
            )
