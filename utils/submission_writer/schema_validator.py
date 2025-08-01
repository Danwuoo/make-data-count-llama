"""Validate submission rows against competition rules."""
from __future__ import annotations

from typing import Iterable

from .schema import SubmissionRow


class SchemaValidator:
    """Ensure rows follow ``id``/``label`` schema and allowed values."""

    allowed_labels = {"primary", "secondary", "none"}

    def validate(self, rows: Iterable[SubmissionRow]) -> None:
        for idx, row in enumerate(rows):
            if not row.id:
                raise ValueError(f"Row {idx} has empty id")
            if row.label not in self.allowed_labels:
                raise ValueError(f"Row {idx} has invalid label {row.label}")
