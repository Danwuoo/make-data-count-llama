from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from .schema import ErrorRecord, ErrorType

try:  # pragma: no cover - optional import
    from utils.llm_inference import LLMResult  # type: ignore
except Exception:  # pragma: no cover
    LLMResult = Any  # type: ignore


class TraceMapper:
    """Utilities to convert trace objects into :class:`ErrorRecord`."""

    @staticmethod
    def from_llm_result(
        result: "LLMResult",
        context_id: str,
        original_label: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        reason: str = "",
    ) -> ErrorRecord:
        meta: Dict[str, Any] = {
            "prompt": getattr(result, "prompt", ""),
            "raw_output": getattr(result, "raw_output", ""),
        }
        return ErrorRecord(
            error_id=(
                f"err_{int(datetime.utcnow().timestamp()*1000)}"
            ),
            context_id=context_id,
            error_type=ErrorType.OTHER,
            source_module="LLMClassifier",
            original_label=original_label,
            predicted_label=getattr(result, "predicted_label", None),
            refined_label=getattr(result, "refined_label", None),
            confidence=getattr(result, "confidence", None),
            confidence_threshold=confidence_threshold,
            reason=reason,
            meta=meta,
        )

    @staticmethod
    def from_correction_proposal(
        proposal: Any, context_id: str, reason: str = ""
    ) -> ErrorRecord:
        meta = {"proposal": proposal}
        return ErrorRecord(
            error_id=(
                f"err_{int(datetime.utcnow().timestamp()*1000)}"
            ),
            context_id=context_id,
            error_type=ErrorType.REFINEMENT_FAILED,
            source_module="SelfCorrector",
            reason=reason or "Correction proposal did not resolve issue",
            meta=meta,
        )
