from __future__ import annotations

from utils.meta_cognition import ErrorLogger, ErrorRecord, ErrorType
from utils.meta_cognition.error_storage import ErrorStorageManager


def test_error_logger_writes_records(tmp_path):
    storage = ErrorStorageManager(base_dir=tmp_path)
    logger = ErrorLogger(storage=storage)

    record = ErrorRecord(
        error_id="err1",
        context_id="ctx1",
        error_type=ErrorType.LOW_CONFIDENCE,
        source_module="LLMClassifier",
        confidence=0.5,
        confidence_threshold=0.7,
        reason="below threshold",
    )

    logger.log(record)

    all_file = tmp_path / "all_errors.jsonl"
    assert all_file.exists()
    lines = all_file.read_text().strip().splitlines()
    assert len(lines) == 1

    loaded = storage.load()
    assert loaded[0].error_id == "err1"

    filtered = storage.load(ErrorType.LOW_CONFIDENCE)
    assert len(filtered) == 1
