from __future__ import annotations

"""Validate ``submission.csv`` files for structural integrity."""
from pathlib import Path
from typing import Iterable, Dict

import pandas as pd

from .column_checker import ColumnStructureChecker
from .row_value_validator import RowValueValidator
from .validator_logger import SchemaValidatorLogger
from .schema import ValidationIssue, ValidationReport


class CSVSchemaValidator:
    """Run a suite of checks against a CSV submission file."""

    def __init__(
        self,
        expected_columns: Iterable[str] | None = None,
        allowed_labels: Iterable[str] | None = None,
        logger: SchemaValidatorLogger | None = None,
    ) -> None:
        self.expected_columns = list(expected_columns or ["id", "label"])
        self.allowed_labels = list(
            allowed_labels or ["primary", "secondary", "none"]
        )
        self.logger = logger or SchemaValidatorLogger()
        self._column_checker = ColumnStructureChecker(self.expected_columns)
        self._row_validator = RowValueValidator(self.allowed_labels)

    def validate(
        self, csv_path: str | Path, report_path: str | None = None
    ) -> ValidationReport:
        """Validate ``csv_path`` and optionally write a JSON report."""
        path = Path(csv_path)
        df = pd.read_csv(path)
        issues: list[ValidationIssue] = []
        error_types: Dict[str, int] = {}

        ok, msg = self._column_checker.check(df.columns)
        if not ok:
            issue = ValidationIssue(
                row_index=None,
                id=None,
                error_type="column_mismatch",
                detail=msg or "",
            )
            self.logger.log_issue(issue)
            issues.append(issue)
            error_types[issue.error_type] = (
                error_types.get(issue.error_type, 0) + 1
            )

        for idx, row in df.iterrows():
            for issue in self._row_validator.validate(idx, row):
                issues.append(issue)
                self.logger.log_issue(issue)
                error_types[issue.error_type] = (
                    error_types.get(issue.error_type, 0) + 1
                )

        error_rows = len(
            {i.row_index for i in issues if i.row_index is not None}
        )
        summary = {
            "total_rows": int(len(df)),
            "valid_rows": int(len(df) - error_rows),
            "errors": int(len(issues)),
            "error_types": error_types,
        }
        report = ValidationReport(
            file=path.name, summary=summary, issues=issues
        )

        if report_path:
            self.logger.write_report(report, report_path)

        return report
