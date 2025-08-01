"""Log prompts and outputs for later replay and analysis."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ReplayRecord:
    """Record stored in the replay log."""

    prompt: str
    output: str
    metadata: Dict[str, Any]


class PromptReplayLogger:
    """Append inference records to a JSONL log file."""

    def __init__(self, path: str | Path = "replay_log.jsonl"):
        self.path = Path(path)
        if not self.path.exists():
            self.path.touch()

    def log(self, record: ReplayRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            json.dump(asdict(record), f)
            f.write("\n")
