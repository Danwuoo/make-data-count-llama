from __future__ import annotations

from typing import Optional

from utils.refinement.schema import CorrectionProposal

from .prompt_assembler import PromptAssembler
from .schema import ContrastivePromptPair, ErrorRecord, PromptPairItem


class PairStrategyEngine:
    """Generate prompt pairs using different strategies."""

    def __init__(
        self,
        mode: str = "direct",
        assembler: Optional[PromptAssembler] = None,
    ) -> None:
        self.mode = mode
        self.assembler = assembler or PromptAssembler()

    def build(
        self, error: ErrorRecord, correction: CorrectionProposal
    ) -> PromptPairItem | ContrastivePromptPair:
        context = error.meta.get("prompt", "")
        question = error.meta.get("question")
        answer = correction.correction_reason

        if self.mode == "direct":
            instruction = (
                "Classify the citation type in the following context."
            )
            input_text = self.assembler.assemble_input(context)
            output = correction.corrected_label
            return PromptPairItem(instruction, input_text, output)

        if self.mode == "qa":
            instruction = (
                "Based on the self-question, determine the correct citation "
                "type."
            )
            input_text = self.assembler.assemble_input(
                context, question, answer
            )
            output = correction.corrected_label
            return PromptPairItem(instruction, input_text, output)

        if self.mode == "contrastive":
            instruction = (
                "Classify the citation type in the following context."
            )
            base_input = self.assembler.assemble_input(context)
            positive = PromptPairItem(
                instruction,
                base_input,
                correction.corrected_label,
            )
            negative = PromptPairItem(
                instruction,
                base_input,
                error.predicted_label or "",
            )
            return ContrastivePromptPair(positive=positive, negative=negative)

        raise ValueError(f"Unknown strategy: {self.mode}")
