from __future__ import annotations

import json
import pandas as pd
import pytest

from utils.submission_writer import KaggleWriter, SubmissionRow
from utils.llm_inference.output_decoder import FinalPrediction


def test_kaggle_writer_creates_csv(tmp_path):
    fp = FinalPrediction(
        context_id="PMC1",
        final_label="Primary",
        confidence=0.9,
        raw_output="",
        used_strategy="",
        label_source="",
        logits={},
    )
    preds = [fp, {"id": "PMC2", "label": " secondary "}]
    writer = KaggleWriter()
    out_csv = tmp_path / "submission.csv"
    writer.write(preds, out_csv)

    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["id", "label"]
    assert df.loc[0, "label"] == "primary"
    assert df.loc[1, "label"] == "secondary"


def test_kaggle_writer_logs_validation_error(tmp_path):
    preds = [SubmissionRow(id="PMC1", label="invalid")]
    writer = KaggleWriter()
    out_csv = tmp_path / "submission.csv"
    report = tmp_path / "report.jsonl"

    with pytest.raises(ValueError):
        writer.write(preds, out_csv, report_path=report)

    assert report.exists()
    lines = report.read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert "invalid" in record["error"]
