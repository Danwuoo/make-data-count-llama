from __future__ import annotations

import json
import pandas as pd

from utils.submission_writer import CSVSchemaValidator


def _write_csv(tmp_path, rows):
    path = tmp_path / "submission.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_validator_passes_valid_file(tmp_path):
    csv_path = _write_csv(tmp_path, [{"id": "A", "label": "primary"}])
    validator = CSVSchemaValidator()
    report = validator.validate(csv_path)
    assert report.summary["errors"] == 0
    assert report.summary["valid_rows"] == 1


def test_validator_reports_issue_and_writes_report(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        [
            {"id": "A", "label": "primary"},
            {"id": "A", "label": "invalid"},
        ],
    )
    validator = CSVSchemaValidator()
    report_path = tmp_path / "report.jsonl"
    report = validator.validate(csv_path, report_path=report_path)
    assert report.summary["errors"] == 2
    assert report.summary["error_types"]["duplicate_id"] == 1
    assert report.summary["error_types"]["invalid_label_value"] == 1
    data = json.loads(report_path.read_text())
    assert data["summary"]["errors"] == 2
