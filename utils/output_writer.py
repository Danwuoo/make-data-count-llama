"""L7: Write predictions into the final Kaggle submission file."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any, List

from .submission_writer import KaggleWriter, CSVSchemaValidator


_writer = KaggleWriter()
_validator = CSVSchemaValidator()


def write_submission(
    rows: Iterable[Any],
    path: str,
    *,
    expected_ids: Iterable[str] | None = None,
    report_path: str | None = None,
) -> None:
    """Write ``rows`` to ``path`` in the competition's submission format."""

    _writer.write(
        rows,
        output_path=path,
        expected_ids=expected_ids,
        report_path=report_path,
    )


def generate_submission(
    predictions_path: str | Path,
    output_path: str | Path,
    *,
    expected_ids: Iterable[str] | None = None,
    report_path: str | None = None,
) -> None:
    """Read predictions from ``predictions_path`` and create a submission.

    The resulting CSV is validated using :class:`CSVSchemaValidator`. If
    validation issues are found a :class:`ValueError` is raised.
    """

    path = Path(predictions_path)
    with path.open("r", encoding="utf-8") as fh:
        predictions: List[Any] = [json.loads(line) for line in fh if line.strip()]

    write_submission(
        predictions,
        str(output_path),
        expected_ids=expected_ids,
        report_path=report_path,
    )

    report = _validator.validate(output_path)

    if report_path:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(report.to_dict()) + "\n")

    if report.summary.get("errors", 0) > 0:
        raise ValueError("submission validation failed")
