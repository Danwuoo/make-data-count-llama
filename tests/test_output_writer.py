from __future__ import annotations

import json
import pandas as pd
import pytest

from utils.output_writer import generate_submission


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_generate_submission_creates_csv_and_report(tmp_path):
    pred_file = tmp_path / "pred.jsonl"
    _write_jsonl(
        pred_file,
        [
            {"id": "A", "label": "primary"},
            {"id": "B", "label": "secondary"},
        ],
    )
    out_csv = tmp_path / "submission.csv"
    report = tmp_path / "report.jsonl"

    generate_submission(pred_file, out_csv, report_path=report)

    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["id", "label"]
    assert df.loc[0, "label"] == "primary"
    assert df.loc[1, "label"] == "secondary"

    lines = report.read_text().strip().splitlines()
    summary = json.loads(lines[-1])
    assert summary["summary"]["errors"] == 0


def test_generate_submission_raises_on_validation_error(tmp_path):
    pred_file = tmp_path / "pred.jsonl"
    _write_jsonl(
        pred_file,
        [
            {"id": "A", "label": "primary"},
            {"id": "A", "label": "secondary"},
        ],
    )
    out_csv = tmp_path / "submission.csv"
    report = tmp_path / "report.jsonl"

    with pytest.raises(ValueError):
        generate_submission(pred_file, out_csv, report_path=report)

    summary = json.loads(report.read_text().strip().splitlines()[-1])
    assert summary["summary"]["errors"] > 0
