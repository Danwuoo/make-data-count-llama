"""L3: Classify citations using LLaMA 3."""
from __future__ import annotations

from typing import Optional

from .llm_inference import LLaMA3Inference, LLMResult

_MODEL_PATH = "models/llama-3-8b-instruct"
_inference: Optional[LLaMA3Inference] = None


def _get_inference() -> Optional[LLaMA3Inference]:
    global _inference
    if _inference is None:
        try:
            _inference = LLaMA3Inference(model_path=_MODEL_PATH)
        except Exception:
            _inference = None
    return _inference


def classify_citation(text: str, context_id: str = "ctx_0") -> str:
    """Classify ``text`` and return the predicted label.

    If the model is unavailable, a dummy label is returned instead.
    """
    inference = _get_inference()
    if inference is None:
        return "primary"
    result: LLMResult = inference.infer(context_id=context_id, context=text)
    return result.predicted_label
