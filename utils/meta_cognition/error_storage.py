from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .schema import ErrorRecord, ErrorType
from config.path_config import ERRORS_DIR


class ErrorStorageManager:
    """Handle persistence of :class:`ErrorRecord` objects."""

    def __init__(self, base_dir: str | Path = ERRORS_DIR) -> None:
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.all_errors_file = self.base_path / "all_errors.jsonl"
        self.file_map = {
            ErrorType.CLASSIFICATION_ERROR: (
                self.base_path / "classification_errors.jsonl"
            ),
            ErrorType.INCONSISTENT_OUTPUT: (
                self.base_path / "unstable_outputs.jsonl"
            ),
            ErrorType.REFINEMENT_FAILED: (
                self.base_path / "refinement_failures.jsonl"
            ),
        }

    def append(self, record: ErrorRecord) -> None:
        """Append ``record`` to storage."""

        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with self.all_errors_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        specific = self.file_map.get(record.error_type)
        if specific is not None:
            with specific.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def load(
        self, error_type: Optional[ErrorType] = None
    ) -> List[ErrorRecord]:
        """Load records from disk.

        When ``error_type`` is ``None`` all errors are loaded. If a specific
        ``error_type`` is given and a dedicated file exists, records are read
        from that file; otherwise the general file is read and records are
        filtered by ``error_type``.
        """

        if error_type is None:
            path = self.all_errors_file
            filter_type: Optional[ErrorType] = None
        else:
            path = self.file_map.get(
                error_type, self.all_errors_file
            )
            filter_type = None if path != self.all_errors_file else error_type

        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            records = [
                ErrorRecord.from_dict(json.loads(line))
                for line in f
                if line.strip()
            ]
        if filter_type is not None:
            records = [r for r in records if r.error_type == filter_type]
        return records
