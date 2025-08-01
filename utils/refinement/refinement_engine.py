from __future__ import annotations

from typing import Dict, List

from utils.llm_inference import LLMResult

from .self_questioner import SelfQuestioner
from .self_corrector import SelfCorrector
from .schema import CorrectionProposal


class RefinementEngine:
    """Coordinate self-questioning and correction steps."""

    def __init__(
        self,
        questioner: SelfQuestioner | None = None,
        corrector: SelfCorrector | None = None,
    ) -> None:
        self.questioner = questioner or SelfQuestioner()
        self.corrector = corrector or SelfCorrector()

    def run(
        self, context_unit: Dict, original_pred: LLMResult
    ) -> List[CorrectionProposal]:
        """Generate questions and attempt corrections."""

        proposals: List[CorrectionProposal] = []
        questions = self.questioner.generate(
            context_unit, original_pred.predicted_label
        )
        for question in questions:
            proposals.append(
                self.corrector.correct(context_unit, question, original_pred)
            )
        return proposals
