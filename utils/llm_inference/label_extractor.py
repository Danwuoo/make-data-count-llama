"""Utilities to extract classification labels from raw LLM text output."""
from __future__ import annotations

from typing import Iterable, Tuple


class LabelExtractor:
    """Extract labels using simple rule-based matching."""

    def __init__(self, labels: Iterable[str] | None = None) -> None:
        self.labels = [lbl.lower() for lbl in (labels or ["primary", "secondary", "none"])]

    def extract(self, text: str) -> Tuple[str, str]:
        """Return a tuple of ``(label, source)`` derived from ``text``.

        The ``source`` explains how the label was obtained:
        ``"direct_label"`` when the entire text equals the label,
        ``"matched_phrase"`` when the label word appears in the text,
        or ``"default"`` if no label could be found.
        """

        lowered = text.strip().lower()
        if lowered in self.labels:
            return lowered, "direct_label"
        for lbl in self.labels:
            if lbl in lowered:
                return lbl, "matched_phrase"
        return "none", "default"
