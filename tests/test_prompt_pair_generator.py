from utils.meta_cognition import ErrorRecord, ErrorType, PromptPairGenerator
from utils.meta_cognition.schema import ContrastivePromptPair, PromptPairItem
from utils.meta_cognition.strategy_engine import PairStrategyEngine
from utils.refinement.schema import CorrectionProposal


def _make_error_and_correction():
    error = ErrorRecord(
        error_id="e1",
        context_id="c1",
        error_type=ErrorType.CLASSIFICATION_ERROR,
        source_module="unit",
        predicted_label="secondary",
        meta={"prompt": "context text", "question": "Is this new data?"},
    )
    correction = CorrectionProposal(
        context_id="c1",
        original_label="secondary",
        corrected_label="primary",
        original_confidence=0.5,
        corrected_confidence=0.9,
        confidence_delta=0.4,
        correction_reason="It references newly generated data.",
        accepted=True,
        question_id="q1",
    )
    return error, correction


def test_generate_direct() -> None:
    error, correction = _make_error_and_correction()
    gen = PromptPairGenerator()
    pairs = gen.generate([error], [correction])
    assert isinstance(pairs[0], PromptPairItem)
    assert pairs[0].output == "primary"


def test_generate_contrastive() -> None:
    error, correction = _make_error_and_correction()
    engine = PairStrategyEngine(mode="contrastive")
    gen = PromptPairGenerator(strategy_engine=engine)
    pairs = gen.generate([error], [correction])
    pair = pairs[0]
    assert isinstance(pair, ContrastivePromptPair)
    assert pair.positive.output == "primary"
    assert pair.negative.output == "secondary"


def test_generate_qa() -> None:
    error, correction = _make_error_and_correction()
    engine = PairStrategyEngine(mode="qa")
    gen = PromptPairGenerator(strategy_engine=engine)
    pairs = gen.generate([error], [correction])
    item = pairs[0]
    assert isinstance(item, PromptPairItem)
    assert "Q:" in item.input
    assert item.output == "primary"
