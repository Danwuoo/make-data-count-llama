"""LLaMA 3 inference backend and backward compatible wrapper."""
from __future__ import annotations

from .base_inference import BaseInferenceModel, LLMResult
from .tokenizer_wrapper import TokenizerWrapper, TokenizerConfig
from .inference_engine import EngineConfig, InferenceEngine


class LLaMA3InferenceModel(BaseInferenceModel):
    """Inference model for LLaMA 3."""

    def load_model(self) -> None:  # pragma: no cover - heavy load
        self.tokenizer = TokenizerWrapper(
            TokenizerConfig(model_path=self.model_path)
        )
        self.engine = InferenceEngine(EngineConfig(model_path=self.model_path))


# Backwards compatibility -----------------------------------------------------
class LLaMA3Inference(LLaMA3InferenceModel):
    """Alias maintaining legacy import path."""

    pass


__all__ = ["LLaMA3InferenceModel", "LLaMA3Inference", "LLMResult"]
