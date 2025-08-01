from __future__ import annotations

"""Logging utilities for :class:`CSVSchemaValidator`."""
import json
from pathlib import Path
from typing import List

from .schema import ValidationIssue, ValidationReport


class SchemaValidatorLogger:
    """Collect and persist validation issues."""

    def __init__(self) -> None:
        self.issues: List[ValidationIssue] = []

    def log_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def write_report(self, report: ValidationReport, path: str) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh)
            fh.write("\n")
