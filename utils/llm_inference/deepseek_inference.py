"""DeepSeek inference backend using HuggingFace APIs."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_inference import BaseInferenceModel


class DeepSeekInferenceModel(BaseInferenceModel):
    """Inference wrapper for DeepSeek models."""

    def load_model(self) -> None:  # pragma: no cover - heavy load
        """Load tokenizer and model from ``self.model_path``."""

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.engine = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )


__all__ = ["DeepSeekInferenceModel"]
