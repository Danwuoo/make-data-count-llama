"""Persist refinement attempts for later auditing."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import CorrectionProposal
from config.path_config import CORRECTIONS_LOG_PATH


class CorrectionLogger:
    """Append correction proposals to a JSONL log file."""

    def __init__(self, log_path: str | Path = CORRECTIONS_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, proposal: CorrectionProposal) -> None:
        """Write ``proposal`` to the JSONL log."""

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(proposal)) + "\n")
