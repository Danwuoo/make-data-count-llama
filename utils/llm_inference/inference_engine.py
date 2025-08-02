"""Core model execution utilities."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM


@dataclass
class EngineConfig:
    """Configuration for the inference engine."""

    model_path: str
    device: str | None = None
    dtype: torch.dtype = torch.float16
    backend: str = "transformers"


class ModelLoader:
    """Load the underlying language model based on backend."""

    def __init__(self, config: EngineConfig):
        self.config = config

    def load(self):  # pragma: no cover - heavy loading
        if self.config.backend == "transformers":
            return AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=self.config.dtype,
                device_map="auto" if self.config.device is None else None,
            )
        raise ValueError(f"Unsupported backend: {self.config.backend}")


class InferenceEngine:
    """Run forward passes on the language model."""

    def __init__(self, config: EngineConfig):
        self.config = config
        loader = ModelLoader(config)
        self.model = loader.load()
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


# ---------------------------------------------------------------------------
# Inference model registry

MODEL_REGISTRY = {
    "llama3": (
        "utils.llm_inference.llama3_inference:LLaMA3InferenceModel"
    ),
    "mixtral": (
        "utils.llm_inference.mixtral_inference:MixtralInferenceModel"
    ),
    "qwen": (
        "utils.llm_inference.qwen_inference:QwenInferenceModel"
    ),
    "gemma": (
        "utils.llm_inference.gemma_inference:GemmaInferenceModel"
    ),
    "deepseek": (
        "utils.llm_inference.deepseek_inference:DeepSeekInferenceModel"
    ),
}


def get_model(model_name: str, *args, **kwargs):
    """Return an inference model instance for ``model_name``."""
    target = MODEL_REGISTRY.get(model_name.lower())
    if not target:
        raise ValueError(f"Unknown model '{model_name}'")
    module_name, class_name = target.split(":")
    module = import_module(module_name)
    model_cls = getattr(module, class_name)
    return model_cls(*args, **kwargs)
