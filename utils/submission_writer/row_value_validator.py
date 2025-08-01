from __future__ import annotations

"""Validate individual submission row values."""
from typing import Iterable, List
import pandas as pd

from .schema import ValidationIssue
from .missing_detector import MissingEntryDetector
from .uniqueness_checker import UniquenessChecker


class RowValueValidator:
    """Check ``id``/``label`` values for each row."""

    def __init__(self, allowed_labels: Iterable[str]) -> None:
        self.allowed = {str(lbl).lower() for lbl in allowed_labels}
        self._missing = MissingEntryDetector()
        self._uniq = UniquenessChecker()

    def validate(
        self, row_index: int, row: pd.Series
    ) -> List[ValidationIssue]:
        issues = self._missing.check(row_index, row)

        if not any(issue.error_type == "missing_label" for issue in issues):
            label = str(row.get("label")).strip().lower()
            if label not in self.allowed:
                allowed = sorted(self.allowed)
                issues.append(
                    ValidationIssue(
                        row_index=row_index,
                        id=row.get("id"),
                        error_type="invalid_label_value",
                        detail=(
                            "Label '{label}' is not one of {allowed}".format(
                                label=row.get("label"),
                                allowed=allowed,
                            )
                        ),
                    )
                )

        if not any(issue.error_type == "missing_id" for issue in issues):
            dup = self._uniq.check(row_index, str(row.get("id")))
            if dup:
                issues.append(dup)

        return issues
