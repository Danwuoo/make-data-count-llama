"""Utilities for meta-cognition error logging."""

from .error_logger import ErrorLogger
from .error_query import ErrorQueryAPI
from .prompt_pair_generator import PromptPairGenerator

try:  # pragma: no cover - optional dependency
    from .lora_fine_tuner import LoRAFineTuner
except Exception:  # noqa: BLE001 - fallback when transformers is missing
    LoRAFineTuner = None
from .schema import (
    ContrastivePromptPair,
    ErrorRecord,
    ErrorSource,
    ErrorType,
    PromptPairItem,
)

__all__ = [
    "ErrorLogger",
    "ErrorQueryAPI",
    "PromptPairGenerator",
    "ErrorRecord",
    "ErrorType",
    "ErrorSource",
    "PromptPairItem",
    "ContrastivePromptPair",
    "LoRAFineTuner",
]
