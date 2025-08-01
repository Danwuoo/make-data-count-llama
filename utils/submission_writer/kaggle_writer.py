"""High-level writer producing Kaggle ``submission.csv`` files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

import pandas as pd

from .submission_formatter import SubmissionFormatter
from .schema_validator import SchemaValidator
from .missing_handler import MissingEntryHandler
from .format_logger import FormatLogger


class KaggleWriter:
    """Convert predictions into a validated Kaggle submission file."""

    def __init__(
        self,
        formatter: SubmissionFormatter | None = None,
        validator: SchemaValidator | None = None,
        missing_handler: MissingEntryHandler | None = None,
        logger: FormatLogger | None = None,
    ) -> None:
        self.formatter = formatter or SubmissionFormatter()
        self.validator = validator or SchemaValidator()
        self.missing_handler = missing_handler or MissingEntryHandler()
        self.logger = logger or FormatLogger()

    def write(
        self,
        predictions: Iterable[Any],
        output_path: str,
        expected_ids: Iterable[str] | None = None,
        report_path: str | None = None,
    ) -> pd.DataFrame:
        """Write ``predictions`` to ``output_path``.

        Parameters
        ----------
        predictions:
            Iterable of prediction objects or dicts containing id/label info.
        output_path:
            Target CSV path.
        expected_ids:
            Optional list of ids that must be present in the final file. Missing
            ids are filled with ``default_label`` in :class:`MissingEntryHandler`.
        report_path:
            If provided, issues encountered are written as JSONL to this path.
        """

        rows = self.formatter.format(predictions)
        rows = self.missing_handler.handle(rows, expected_ids)

        try:
            self.validator.validate(rows)
        except ValueError as exc:  # pragma: no cover - validation errors are rare
            self.logger.log(error=str(exc))
            if report_path:
                self.logger.write_report(report_path)
            raise

        data = [{"id": r.id, "label": r.label} for r in rows]
        df = pd.DataFrame(data, columns=["id", "label"])

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

        if report_path and self.logger.issues:
            self.logger.write_report(report_path)

        return df
