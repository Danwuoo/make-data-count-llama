from __future__ import annotations

"""Detect missing fields in submission rows."""
from typing import List
import pandas as pd

from .schema import ValidationIssue


class MissingEntryDetector:
    """Identify empty ``id`` or ``label`` fields."""

    def check(self, row_index: int, row: pd.Series) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        row_id = row.get("id")
        label = row.get("label")
        if pd.isna(row_id) or row_id == "":
            issues.append(
                ValidationIssue(
                    row_index=row_index,
                    id=None,
                    error_type="missing_id",
                    detail="Empty id",
                )
            )
        if pd.isna(label) or label == "":
            issues.append(
                ValidationIssue(
                    row_index=row_index,
                    id=row_id if not pd.isna(row_id) else None,
                    error_type="missing_label",
                    detail="Label is missing",
                )
            )
        return issues
