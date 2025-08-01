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
