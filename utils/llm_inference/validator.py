"""Validation utilities for LLaMA3 inference results."""
from __future__ import annotations

from typing import Dict

LABELS = {"primary", "secondary", "none"}


class InferenceValidator:
    """Simple validator ensuring outputs are well-formed."""

    def validate(self, result: Dict[str, object]) -> None:
        label = result.get("predicted_label")
        if label not in LABELS:
            raise ValueError(f"Invalid label: {label}")
