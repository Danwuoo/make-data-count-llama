"""Locate semantic zones to improve precision."""

from __future__ import annotations

from typing import Iterable, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from .doi_recognizer import StructuredID


class SemanticZoneLocator:
    """Filter identifiers based on contextual metadata."""

    def filter(self, ids: Iterable["StructuredID"], meta: dict) -> List["StructuredID"]:
        section = meta.get("section")
        if not section:
            return list(ids)
        allowed = {"abstract", "body", "methods", "references"}
        if section.lower() in allowed:
            return list(ids)
        return []
