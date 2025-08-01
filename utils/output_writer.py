"""L7: Write predictions into the final Kaggle submission file."""
from __future__ import annotations

from typing import Iterable, Any

from .submission_writer import KaggleWriter


_writer = KaggleWriter()


def write_submission(
    rows: Iterable[Any], path: str, *, expected_ids: Iterable[str] | None = None, report_path: str | None = None
) -> None:
    """Write ``rows`` to ``path`` in the competition's submission format.

    Parameters
    ----------
    rows:
        Iterable of prediction objects or dictionaries containing id/label
        information.
    path:
        Destination CSV file path.
    expected_ids:
        Optional iterable of ids that must be present in the submission. Missing
        ids are filled using the default label.
    report_path:
        Optional path to a JSON Lines file where validation issues are written.
    """

    _writer.write(rows, output_path=path, expected_ids=expected_ids, report_path=report_path)
