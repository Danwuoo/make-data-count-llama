"""Qwen model inference backend using HuggingFace Transformers."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_inference import BaseInferenceModel


class QwenInferenceModel(BaseInferenceModel):
    """Inference model for Qwen family of models."""

    def load_model(self) -> None:  # pragma: no cover - heavy load
        """Load the tokenizer and model from ``self.model_path``."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.engine = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )


__all__ = ["QwenInferenceModel"]
