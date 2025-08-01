"""Apply metadata-based penalties or boosts to context scores."""

from __future__ import annotations

from typing import Any, Dict


class ContextPenaltyRule:
    """Score adjustment based on metadata fields.

    Each field in ``penalties`` maps to a dictionary of value -> adjustment.
    Adjustments may be negative (penalty) or positive (boost).
    """

    def __init__(self, penalties: Dict[str, Dict[Any, float]] | None = None) -> None:
        self.penalties = penalties or {}

    def apply(self, metadata: Dict[str, Any]) -> float:
        """Return the total adjustment for the provided ``metadata``."""

        adjustment = 0.0
        for field, mapping in self.penalties.items():
            value = metadata.get(field)
            if value in mapping:
                adjustment += mapping[value]
        return adjustment
