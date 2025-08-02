"""Mixtral mixture-of-experts inference backend."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_inference import BaseInferenceModel


class MixtralInferenceModel(BaseInferenceModel):
    """Inference model for Mixtral-style MoE models."""

    def load_model(self) -> None:  # pragma: no cover - heavy load
        """Load tokenizer and engine for Mixtral models.

        Models are loaded from ``self.model_path`` using HuggingFace's
        :func:`~transformers.AutoTokenizer.from_pretrained` and
        :func:`~transformers.AutoModelForCausalLM.from_pretrained` helpers. The
        model weights are cast to ``float16`` and the device placement is
        automatically determined. If the model configuration supports Flash
        Attention v2, it is enabled to improve inference speed.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.engine = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        if hasattr(self.engine, "config") and hasattr(
            self.engine.config, "attn_implementation"
        ):
            self.engine.config.attn_implementation = "flash_attention_2"


__all__ = ["MixtralInferenceModel"]
