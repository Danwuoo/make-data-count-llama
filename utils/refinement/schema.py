from dataclasses import dataclass
from typing import Dict


@dataclass
class SelfQuestionItem:
    """Structured representation of a self-generated question."""

    context_id: str
    question_id: str
    question_text: str
    question_type: str
    confidence_level: float
    source: Dict[str, str]


@dataclass
class CorrectionProposal:
    """Structured output describing a correction attempt."""

    context_id: str
    original_label: str
    corrected_label: str
    original_confidence: float
    corrected_confidence: float
    confidence_delta: float
    correction_reason: str
    accepted: bool
    question_id: str
    metadata: Dict[str, str | float] | None = None


@dataclass
class CorrectionDelta:
    """Difference metrics between original and corrected prediction."""

    original_label: str
    corrected_label: str
    original_confidence: float
    corrected_confidence: float

    @property
    def confidence_delta(self) -> float:  # pragma: no cover - simple property
        return self.corrected_confidence - self.original_confidence
