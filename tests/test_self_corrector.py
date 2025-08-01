from dataclasses import dataclass

from utils.refinement import SelfCorrector
from utils.refinement.correction_engine import CorrectionEngine
from utils.refinement.schema import CorrectionProposal, SelfQuestionItem


@dataclass
class DummyResult:
    context_id: str
    predicted_label: str
    confidence: float
    raw_output: str
    prompt: str
    logits: dict
    meta: dict


class DummyInference:
    """Simple stub returning a fixed label and confidence."""

    def infer(self, context_id: str, context: str) -> DummyResult:  # type: ignore[override]
        return DummyResult(
            context_id=context_id,
            predicted_label="primary",
            confidence=0.9,
            raw_output="",  # pragma: no cover - unused fields
            prompt=context,
            logits={},
            meta={},
        )


def test_self_corrector_accepts_change() -> None:
    engine = CorrectionEngine(inference=DummyInference())
    corrector = SelfCorrector(engine=engine)

    context = {"context_id": "ctx1", "text": "Context text"}
    question = SelfQuestionItem(
        context_id="ctx1",
        question_id="q1",
        question_text="Does this refer to new data?",
        question_type="clarify",
        confidence_level=0.5,
        source={},
    )
    original = DummyResult(
        context_id="ctx1",
        predicted_label="secondary",
        confidence=0.6,
        raw_output="",
        prompt="",
        logits={},
        meta={},
    )
    proposal = corrector.correct(context, question, original)

    assert isinstance(proposal, CorrectionProposal)
    assert proposal.accepted is True
    assert proposal.corrected_label == "primary"
    assert proposal.original_label == "secondary"
