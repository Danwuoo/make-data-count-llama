from utils.meta_loop import run_meta_loop
from utils.meta_cognition import ErrorRecord, ErrorType
from utils.refinement.schema import CorrectionProposal


def make_error_and_correction():
    error = ErrorRecord(
        error_id="e1",
        context_id="c1",
        error_type=ErrorType.CLASSIFICATION_ERROR,
        source_module="unit",
        predicted_label="secondary",
        meta={"prompt": "context", "question": "Is this new data?"},
    )
    correction = CorrectionProposal(
        context_id="c1",
        original_label="secondary",
        corrected_label="primary",
        original_confidence=0.5,
        corrected_confidence=0.9,
        confidence_delta=0.4,
        correction_reason="It references new data.",
        accepted=True,
        question_id="q1",
    )
    return error, correction


def test_run_meta_loop_logs_and_generates(tmp_path):
    error, correction = make_error_and_correction()
    pairs, metrics = run_meta_loop(
        [error],
        [correction],
        model=None,
        tokenizer=None,
        error_dir=tmp_path / "errors",
        adapter_dir=tmp_path / "adapters",
    )
    assert pairs[0].output == "primary"
    assert metrics == {}
    log_file = tmp_path / "errors" / "all_errors.jsonl"
    assert log_file.exists()
