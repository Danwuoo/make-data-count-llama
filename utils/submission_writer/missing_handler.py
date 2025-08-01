"""Utilities for handling missing submission entries."""
from __future__ import annotations

from typing import Iterable, List, Optional

from .schema import SubmissionRow


class MissingEntryHandler:
    """Fill in missing IDs with a default label."""

    def handle(
        self,
        rows: List[SubmissionRow],
        expected_ids: Optional[Iterable[str]] = None,
        default_label: str = "none",
    ) -> List[SubmissionRow]:
        if not expected_ids:
            return rows
        existing = {row.id for row in rows}
        for expected in expected_ids:
            if expected not in existing:
                rows.append(SubmissionRow(id=str(expected), label=default_label))
        return rows
