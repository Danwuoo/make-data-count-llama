"""Core model execution utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM


@dataclass
class EngineConfig:
    """Configuration for the inference engine."""

    model_path: str
    device: str | None = None
    dtype: torch.dtype = torch.float16


class InferenceEngine:
    """Run forward passes on the language model."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=config.dtype,
            device_map="auto" if config.device is None else None,
        )
        if config.device:
            self.model.to(config.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        inputs: Dict[str, Any],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate text and collect logits for the last token."""
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=True,
        )
        tokens = output.sequences[0]
        scores = output.scores
        return {
            "tokens": tokens,
            "scores": scores,
        }
