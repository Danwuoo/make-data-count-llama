from dataclasses import dataclass

from utils.refinement import RefinementEngine, SelfCorrector, SelfQuestioner
from utils.refinement.correction_engine import CorrectionEngine
from utils.refinement.schema import CorrectionProposal


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
    def infer(self, context_id: str, context: str) -> DummyResult:  # type: ignore[override]
        return DummyResult(
            context_id=context_id,
            predicted_label="primary",
            confidence=0.9,
            raw_output="",
            prompt=context,
            logits={},
            meta={},
        )


def test_refinement_engine_flow() -> None:
    corrector = SelfCorrector(engine=CorrectionEngine(inference=DummyInference()))
    engine = RefinementEngine(questioner=SelfQuestioner(), corrector=corrector)

    context = {"context_id": "ctx1", "text": "This refers to the XYZ dataset."}
    original = DummyResult(
        context_id="ctx1",
        predicted_label="secondary",
        confidence=0.6,
        raw_output="",
        prompt="",
        logits={},
        meta={},
    )

    proposals = engine.run(context, original)

    assert proposals, "No proposals produced"
    assert isinstance(proposals[0], CorrectionProposal)
    assert proposals[0].accepted is True
    assert proposals[0].corrected_label == "primary"
