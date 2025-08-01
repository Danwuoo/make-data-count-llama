from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class ErrorType(str, Enum):
    """Supported error categories for logging."""

    CLASSIFICATION_ERROR = "classification_error"
    LOW_CONFIDENCE = "low_confidence"
    INCONSISTENT_OUTPUT = "inconsistent_output"
    REFINEMENT_FAILED = "refinement_failed"
    OUTPUT_DECODER_ERROR = "output_decoder_error"
    OTHER = "other"


class ErrorSource(str, Enum):
    """Source modules that can emit errors."""

    LLM_CLASSIFIER = "LLMClassifier"
    REFINEMENT_ENGINE = "RefinementEngine"
    PROMPT_PERTURBATION_TESTER = "PromptPerturbationTester"
    SELF_CORRECTOR = "SelfCorrector"
    DECODER = "LLMOutputDecoder"
    OTHER = "other"


@dataclass
class ErrorRecord:
    """Structured representation of a logged error."""

    error_id: str
    context_id: str
    error_type: ErrorType
    source_module: str
    original_label: Optional[str] = None
    predicted_label: Optional[str] = None
    refined_label: Optional[str] = None
    confidence: Optional[float] = None
    confidence_threshold: Optional[float] = None
    reason: str = ""
    timestamp: str = field(
        default_factory=lambda: (
            datetime.utcnow().isoformat(timespec="seconds") + "Z"
        )
    )
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["error_type"] = self.error_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorRecord":
        data = data.copy()
        data["error_type"] = ErrorType(data["error_type"])
        return cls(**data)
