"""Reusable abstractions for model-specific inference backends.

This module defines the :class:`BaseInferenceModel` used by all inference
wrappers (e.g., LLaMA 3, Mixtral, Qwen, Gemma, DeepSeek). It provides common
components such as prompt generation, output decoding, validation and optional
prompt replay logging.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .output_decoder import LLMOutputDecoder, DecodingStrategy
from .prompt_generator import PromptGenerator
from .replay_logger import PromptReplayLogger, ReplayRecord
from .validator import InferenceValidator


@dataclass
class LLMResult:
    """Structured result returned by inference models."""

    context_id: str
    predicted_label: str
    confidence: float
    raw_output: str
    prompt: str
    logits: Dict[str, float]
    meta: Dict[str, Any]


class BaseInferenceModel(ABC):
    """Abstract base class for all inference backends."""

    def __init__(
        self,
        model_path: str,
        replay_log: Optional[str] = None,
        template_version: str = "v1.0",
    ) -> None:
        self.model_path = model_path
        self.template_version = template_version
        self.prompt_generator = PromptGenerator()
        self.decoder = LLMOutputDecoder()
        self.validator = InferenceValidator()
        self.logger: Optional[PromptReplayLogger] = (
            PromptReplayLogger(replay_log) if replay_log else None
        )
        self.model_name = model_path.split("/")[-1]
        self.load_model()

    def load_model(self) -> None:  # pragma: no cover - heavy load
        """Instantiate tokenizer and engine for the model.

        This default implementation loads a HuggingFace-compatible model and
        tokenizer from ``self.model_path``. Subclasses may override for models
        requiring special handling (e.g., mixture-of-experts).
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

    def format_prompt(self, context: str, strategy: str) -> str:
        """Format the prompt for the given ``context`` and ``strategy``."""
        return self.prompt_generator.generate(context, strategy)

    def predict(
        self,
        context_id: str,
        context: str,
        strategy: str = "zero-shot",
        temperature: float = 0.0,
        max_new_tokens: int = 32,
    ) -> LLMResult:
        """Run inference on ``context`` and return an :class:`LLMResult`."""
        prompt = self.format_prompt(context, strategy)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.engine.device
        )
        outputs = self.engine.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            output_scores=True,
            return_dict_in_generate=True,
        )
        text = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )
        scores = [score[0].tolist() for score in outputs.scores]
        prediction = self.decoder.decode(
            context_id=context_id,
            text=text,
            scores=scores,
            strategy=DecodingStrategy.TEXT2LABEL,
        )
        result = LLMResult(
            context_id=context_id,
            predicted_label=prediction.final_label,
            confidence=prediction.confidence,
            raw_output=prediction.raw_output,
            prompt=prompt,
            logits=prediction.logits,
            meta={
                "model_name": self.model_name,
                "template_version": self.template_version,
                "temperature": temperature,
                "used_strategy": prediction.used_strategy,
                "label_source": prediction.label_source,
            },
        )
        self.validator.validate(asdict(result))
        if self.logger:
            self.logger.log(
                ReplayRecord(
                    prompt=prompt,
                    output=prediction.raw_output,
                    metadata=result.meta,
                )
            )
        return result

    # Backwards compatibility for older code using ``infer``
    def infer(self, *args: Any, **kwargs: Any) -> LLMResult:
        return self.predict(*args, **kwargs)


# ---------------------------------------------------------------------------
# Model registry and loader utilities

from .llama3_inference import LLaMA3InferenceModel  # noqa: E402
from .qwen_inference import QwenInferenceModel  # noqa: E402
from .deepseek_inference import DeepSeekInferenceModel  # noqa: E402
from .mixtral_inference import MixtralInferenceModel  # noqa: E402
from .gemma_inference import GemmaInferenceModel  # noqa: E402


MODEL_REGISTRY = {
    "llama3": LLaMA3InferenceModel,
    "qwen": QwenInferenceModel,
    "deepseek": DeepSeekInferenceModel,
    "mixtral": MixtralInferenceModel,
    "gemma": GemmaInferenceModel,
}


def get_inference_model(
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    **kwargs: Any,
) -> BaseInferenceModel:
    """Instantiate an inference model based on ``model_name`` or ``model_path``."""

    if model_name:
        cls = MODEL_REGISTRY.get(model_name.lower())
        if not cls:
            raise ValueError(f"Unsupported model name: {model_name}")
        return cls(model_path=model_path or model_name, **kwargs)
    if model_path:
        lowered = model_path.lower()
        for key, cls in MODEL_REGISTRY.items():
            if key in lowered:
                return cls(model_path=model_path, **kwargs)
        raise ValueError(f"Could not auto-detect model type from path: {model_path}")
    raise ValueError("You must provide either model_name or model_path")


__all__ = [
    "BaseInferenceModel",
    "LLMResult",
    "MODEL_REGISTRY",
    "get_inference_model",
]
