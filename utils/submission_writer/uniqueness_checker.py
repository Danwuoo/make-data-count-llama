from __future__ import annotations

"""Detect duplicate identifiers in submission rows."""
from typing import Optional

from .schema import ValidationIssue


class UniquenessChecker:
    """Track seen ids and flag duplicates."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def check(self, row_index: int, row_id: str) -> Optional[ValidationIssue]:
        if row_id in self._seen:
            return ValidationIssue(
                row_index=row_index,
                id=row_id,
                error_type="duplicate_id",
                detail=f"Duplicate id {row_id}",
            )
        self._seen.add(row_id)
        return None
