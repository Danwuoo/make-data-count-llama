"""Core module executing the self-correction loop."""

from __future__ import annotations

from typing import Dict

from utils.llm_inference import LLMResult

from .change_detector import ChangeDetector
from .correction_engine import CorrectionEngine
from .correction_logger import CorrectionLogger
from .reask_prompt_generator import ReAskPromptGenerator
from .schema import CorrectionProposal, SelfQuestionItem


class SelfCorrector:
    """Trigger a re-ask and decide whether to accept the new label."""

    def __init__(
        self,
        prompt_generator: ReAskPromptGenerator | None = None,
        engine: CorrectionEngine | None = None,
        detector: ChangeDetector | None = None,
        logger: CorrectionLogger | None = None,
    ) -> None:
        self.prompt_generator = prompt_generator or ReAskPromptGenerator()
        self.engine = engine or CorrectionEngine()
        self.detector = detector or ChangeDetector()
        self.logger = logger

    def correct(
        self,
        context_unit: Dict,
        self_question: SelfQuestionItem,
        original_pred: LLMResult,
    ) -> CorrectionProposal:
        """Run the correction flow for a single question."""

        context_text = context_unit.get("text", "")
        prompt = self.prompt_generator.build(
            context=context_text, question=self_question.question_text
        )
        result = self.engine.run(context_id=self_question.context_id, prompt=prompt)
        accepted, reason = self.detector.evaluate(
            original_label=original_pred.predicted_label,
            original_confidence=original_pred.confidence,
            corrected_label=result.predicted_label,
            corrected_confidence=result.confidence,
        )
        metadata = {
            "original_confidence": original_pred.confidence,
            "corrected_confidence": result.confidence,
            "reask_prompt": prompt,
            "raw_response": getattr(result, "raw_output", ""),
        }
        proposal = CorrectionProposal(
            context_id=self_question.context_id,
            original_label=original_pred.predicted_label,
            corrected_label=result.predicted_label,
            original_confidence=original_pred.confidence,
            corrected_confidence=result.confidence,
            confidence_delta=result.confidence - original_pred.confidence,
            correction_reason=reason,
            accepted=accepted,
            question_id=self_question.question_id,
            metadata=metadata,
        )
        if self.logger:
            self.logger.log(proposal)
        return proposal
