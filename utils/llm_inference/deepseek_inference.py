"""Placeholder DeepSeek inference backend."""
from __future__ import annotations

from .base_inference import BaseInferenceModel


class DeepSeekInferenceModel(BaseInferenceModel):
    """DeepSeek model support is not implemented yet."""

    def load_model(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError(
            "DeepSeekInferenceModel is not implemented yet."
        )


__all__ = ["DeepSeekInferenceModel"]
