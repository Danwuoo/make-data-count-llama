"""High level wrapper providing LLaMA 3 inference services."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .inference_engine import EngineConfig, InferenceEngine
from .output_parser import OutputParser
from .prompt_generator import PromptGenerator
from .replay_logger import PromptReplayLogger, ReplayRecord
from .tokenizer_wrapper import TokenizerConfig, TokenizerWrapper
from .validator import InferenceValidator


@dataclass
class LLMResult:
    """Structured result returned by :class:`LLaMA3Inference`."""

    context_id: str
    predicted_label: str
    confidence: float
    raw_output: str
    prompt: str
    logits: Dict[str, float]
    meta: Dict[str, Any]


class LLaMA3Inference:
    """Encapsulates prompt construction, inference and parsing."""

    def __init__(
        self,
        model_path: str,
        replay_log: str | None = None,
        template_version: str = "v1.0",
    ) -> None:
        self.prompt_generator = PromptGenerator()
        self.tokenizer = TokenizerWrapper(
            TokenizerConfig(model_path=model_path)
        )
        self.engine = InferenceEngine(EngineConfig(model_path=model_path))
        self.parser = OutputParser()
        self.validator = InferenceValidator()
        self.logger: Optional[PromptReplayLogger] = None
        if replay_log:
            self.logger = PromptReplayLogger(replay_log)
        self.template_version = template_version
        self.model_name = model_path.split("/")[-1]

    def infer(
        self,
        context_id: str,
        context: str,
        strategy: str = "zero-shot",
        temperature: float = 0.0,
        max_new_tokens: int = 32,
    ) -> LLMResult:
        """Run inference on ``context`` and return an :class:`LLMResult`."""
        prompt = self.prompt_generator.generate(context, strategy)
        tokens = self.tokenizer.encode(prompt)
        generated = self.engine.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        text = self.tokenizer.decode(generated["tokens"])
        parsed = self.parser.parse(text, generated["scores"])

        result = LLMResult(
            context_id=context_id,
            predicted_label=parsed.label,
            confidence=parsed.confidence,
            raw_output=parsed.text,
            prompt=prompt,
            logits=parsed.logits,
            meta={
                "model_name": self.model_name,
                "template_version": self.template_version,
                "temperature": temperature,
            },
        )
        self.validator.validate(asdict(result))

        if self.logger:
            self.logger.log(
                ReplayRecord(
                    prompt=prompt,
                    output=parsed.text,
                    metadata=result.meta,
                )
            )
        return result
