"""Heuristic context based scoring for accession identifiers."""

from __future__ import annotations

from typing import Optional

from .pattern_registry import PatternMatch


class ContextWeighter:
    """Assign a confidence score based on local textual context."""

    KEYWORDS = {"data", "dataset", "accession", "geo", "sra"}
    PRIORITY_SECTIONS = {"methods", "data availability"}

    def weight(
        self, text: str, match: PatternMatch, meta: Optional[dict] = None
    ) -> float:
        meta = meta or {}
        score = 0.5
        start = max(0, match.start - 50)
        end = match.end + 50
        window = text[start:end].lower()
        if any(k in window for k in self.KEYWORDS):
            score += 0.3
        section = str(meta.get("section", "")).lower()
        if section in self.PRIORITY_SECTIONS:
            score += 0.2
        return min(score, 1.0)


__all__ = ["ContextWeighter"]
