from __future__ import annotations

from .schema import ErrorRecord, ErrorType


class ErrorClassifier:
    """Infer an :class:`ErrorType` from an :class:`ErrorRecord`."""

    def classify(self, record: ErrorRecord) -> ErrorType:
        if record.original_label and record.predicted_label:
            if record.original_label != record.predicted_label:
                return ErrorType.CLASSIFICATION_ERROR
        if (
            record.confidence is not None
            and record.confidence_threshold is not None
            and record.confidence < record.confidence_threshold
        ):
            return ErrorType.LOW_CONFIDENCE
        if record.meta.get("is_inconsistent"):
            return ErrorType.INCONSISTENT_OUTPUT
        if record.meta.get("refinement_failed"):
            return ErrorType.REFINEMENT_FAILED
        if record.meta.get("decoder_error"):
            return ErrorType.OUTPUT_DECODER_ERROR
        return ErrorType.OTHER
