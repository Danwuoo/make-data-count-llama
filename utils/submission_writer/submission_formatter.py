"""Convert various prediction structures into :class:`SubmissionRow`."""
from __future__ import annotations

from typing import Iterable, Any, List

from .schema import SubmissionRow


class SubmissionFormatter:
    """Format raw predictions into ``SubmissionRow`` items."""

    def format(self, predictions: Iterable[Any]) -> List[SubmissionRow]:
        rows: List[SubmissionRow] = []
        for pred in predictions:
            row_id: str | None = None
            label: str | None = None

            if isinstance(pred, SubmissionRow):
                row_id = pred.id
                label = pred.label
            elif hasattr(pred, "context_id") and hasattr(pred, "final_label"):
                row_id = getattr(pred, "context_id")
                label = getattr(pred, "final_label")
            elif isinstance(pred, dict):
                row_id = pred.get("id") or pred.get("context_id")
                label = pred.get("label") or pred.get("final_label")

            if row_id is None or label is None:
                continue

            rows.append(
                SubmissionRow(id=str(row_id).strip(), label=str(label).strip().lower())
            )
        return rows
