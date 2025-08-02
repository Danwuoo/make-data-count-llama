"""Base classes for model-specific inference wrappers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

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

    @abstractmethod
    def load_model(self) -> None:
        """Instantiate tokenizer and engine for the model."""

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
        tokens = self.tokenizer.encode(prompt)
        generated = self.engine.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        text = self.tokenizer.decode(generated["tokens"])
        prediction = self.decoder.decode(
            context_id=context_id,
            text=text,
            scores=generated["scores"],
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
