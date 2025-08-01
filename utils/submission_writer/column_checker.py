from __future__ import annotations

"""Column structure validation for submission files."""
from typing import Sequence, Tuple


class ColumnStructureChecker:
    """Ensure required columns exist in the expected order."""

    def __init__(self, expected: Sequence[str]) -> None:
        self.expected = list(expected)

    def check(self, columns: Sequence[str]) -> Tuple[bool, str | None]:
        cols = list(columns)
        if cols != self.expected:
            return False, f"Expected columns {self.expected} but found {cols}"
        return True, None
