"""Schema definitions for Kaggle submission rows and validation reports."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SubmissionRow:
    """Single row in the final submission file."""

    id: str
    label: str


@dataclass
class ValidationIssue:
    """A single validation problem discovered in ``submission.csv``."""

    row_index: Optional[int]
    id: Optional[str]
    error_type: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Summary of validation results for a submission file."""

    file: str
    summary: Dict[str, Any]
    issues: List[ValidationIssue]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "summary": self.summary,
            "issues": [issue.to_dict() for issue in self.issues],
        }
