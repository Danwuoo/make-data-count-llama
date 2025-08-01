from utils.llm_inference.output_decoder import (
    LLMOutputDecoder,
    DecodingStrategy,
)


def _make_scores(mapping):
    """Create scores list with logits according to ``mapping`` label->logit."""
    vocab = [0.0] * 10000
    for label, logit in mapping.items():
        idx = abs(hash(label)) % 10000
        vocab[idx] = logit
    return [vocab]


def test_decode_text2label():
    decoder = LLMOutputDecoder()
    scores = _make_scores({"primary": 2.0, "secondary": 1.0, "none": -1.0})
    prediction = decoder.decode(
        context_id="ctx_test",
        text="This is a primary citation.",
        scores=scores,
        strategy=DecodingStrategy.TEXT2LABEL,
    )
    assert prediction.final_label == "primary"
    assert prediction.used_strategy == "text2label"
    assert "primary" in prediction.logits


def test_decode_logit_mapped():
    decoder = LLMOutputDecoder()
    scores = _make_scores({"primary": -1.0, "secondary": 3.0, "none": 0.0})
    prediction = decoder.decode(
        context_id="ctx_test",
        text="No explicit label here.",
        scores=scores,
        strategy=DecodingStrategy.LOGIT_MAPPED,
    )
    assert prediction.final_label == "secondary"
    assert prediction.label_source == "logit"
