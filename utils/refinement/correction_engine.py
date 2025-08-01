"""Wrapper around :class:`LLaMA3Inference` for re-asking."""

from __future__ import annotations

from utils.llm_inference import LLaMA3Inference, LLMResult


class CorrectionEngine:
    """Trigger a second reasoning pass using an inference engine."""

    def __init__(self, inference: LLaMA3Inference | None = None) -> None:
        self.inference = inference or LLaMA3Inference(model_path="/tmp/llama")

    def run(self, context_id: str, prompt: str) -> LLMResult:
        """Run inference on the constructed prompt."""

        return self.inference.infer(context_id=context_id, context=prompt)
