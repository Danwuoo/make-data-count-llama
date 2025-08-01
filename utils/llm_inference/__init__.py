"""LLM inference utilities for LLaMA 3."""

from .output_decoder import LLMOutputDecoder, FinalPrediction, DecodingStrategy

try:  # pragma: no cover - optional heavy imports
    from .llama3_inference import LLaMA3Inference, LLMResult
except Exception:  # pragma: no cover - torch or other deps missing
    LLaMA3Inference = None  # type: ignore
    LLMResult = None  # type: ignore

__all__ = [
    "LLaMA3Inference",
    "LLMResult",
    "LLMOutputDecoder",
    "FinalPrediction",
    "DecodingStrategy",
]
