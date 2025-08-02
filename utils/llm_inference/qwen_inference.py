"""Placeholder Qwen inference backend."""
from __future__ import annotations

from .base_inference import BaseInferenceModel


class QwenInferenceModel(BaseInferenceModel):
    """Qwen model support is not implemented yet."""

    def load_model(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError(
            "QwenInferenceModel is not implemented yet."
        )


__all__ = ["QwenInferenceModel"]
