"""L3: High level classifier orchestrating inference and perturbation tests."""
from __future__ import annotations

from typing import Optional

from .meta_cognition import ErrorLogger
from .meta_cognition.trace_mapper import TraceMapper

from .llm_inference import (
    LLaMA3Inference,
    LLMResult,
    FinalPrediction,
    PromptPerturbationTester,
    PerturbationReport,
)

_MODEL_PATH = "models/llama-3-8b-instruct"


class LLMClassifier:
    """Controller that runs inference and optional robustness checks."""

    def __init__(
        self,
        inference: Optional[LLaMA3Inference] = None,
        tester: Optional[PromptPerturbationTester] = None,
    ) -> None:
        self.inference = inference or LLaMA3Inference(model_path=_MODEL_PATH)
        self.tester = tester

    def classify(
        self,
        text: str,
        context_id: str = "ctx_0",
        strategy: str = "zero-shot",
        run_perturbation: bool | None = None,
        *,
        error_logger: ErrorLogger | None = None,
        confidence_threshold: float | None = None,
    ) -> tuple[FinalPrediction, list[str]]:
        """Classify ``text`` and return prediction and logged error IDs.

        When ``run_perturbation`` is ``True`` (default when a tester is
        provided), the classifier performs prompt perturbation testing and
        annotates the prediction with consistency metrics. If ``error_logger``
        and ``confidence_threshold`` are supplied, predictions whose
        confidence falls below ``confidence_threshold`` are logged as
        :class:`~utils.meta_cognition.schema.ErrorRecord` instances and the
        corresponding error IDs are returned.
        """

        result: LLMResult = self.inference.infer(
            context_id=context_id, context=text, strategy=strategy
        )
        prediction = FinalPrediction(
            context_id=context_id,
            final_label=result.predicted_label,
            confidence=result.confidence,
            raw_output=result.raw_output,
            used_strategy=result.meta.get("used_strategy", ""),
            label_source=result.meta.get("label_source", ""),
            logits=result.logits,
        )

        should_run = (
            run_perturbation
            if run_perturbation is not None
            else self.tester is not None
        )
        if should_run and self.tester is not None:
            report: PerturbationReport = self.tester.test(
                context_id=context_id,
                prompt=result.prompt,
                original_label=result.predicted_label,
                original_confidence=result.confidence,
            )
            prediction.is_consistent = report.is_consistent
            prediction.perturbation_score = report.invariance_score

        error_ids: list[str] = []
        if (
            error_logger is not None
            and confidence_threshold is not None
            and result.confidence < confidence_threshold
        ):
            record = TraceMapper.from_llm_result(
                result,
                context_id=context_id,
                confidence_threshold=confidence_threshold,
                reason="confidence below threshold",
            )
            logged = error_logger.log(record)
            error_ids.append(logged.error_id)

        return prediction, error_ids


_classifier: Optional[LLMClassifier] = None


def _get_classifier() -> Optional[LLMClassifier]:
    """Return a cached :class:`LLMClassifier` instance."""
    global _classifier
    if _classifier is None:
        try:
            inference = LLaMA3Inference(model_path=_MODEL_PATH)
            tester: Optional[PromptPerturbationTester] = None
            _classifier = LLMClassifier(inference=inference, tester=tester)
        except Exception:
            _classifier = None
    return _classifier


def classify_citation(text: str, context_id: str = "ctx_0") -> str:
    """Classify ``text`` and return the predicted label.

    If model loading fails, a dummy label is returned.
    """
    classifier = _get_classifier()
    if classifier is None:
        return "primary"
    prediction, _ = classifier.classify(text=text, context_id=context_id)
    return prediction.final_label
