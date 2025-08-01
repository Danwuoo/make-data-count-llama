"""Logging utilities for submission formatting and validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class FormatLogger:
    """Collect issues encountered during formatting/validation."""

    def __init__(self) -> None:
        self.issues: List[Dict[str, Any]] = []

    def log(self, **info: Any) -> None:
        self.issues.append(info)

    def write_report(self, path: str) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as fh:
            for issue in self.issues:
                fh.write(json.dumps(issue) + "\n")
