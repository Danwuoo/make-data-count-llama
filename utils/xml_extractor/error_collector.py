"""Collect errors during XML parsing."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


class XMLErrorCollector:
    """Simple error accumulator."""

    def __init__(self) -> None:
        self.errors: Dict[str, List[str]] = defaultdict(list)

    def add(self, stage: str, exc: Exception) -> None:
        self.errors[stage].append(str(exc))
