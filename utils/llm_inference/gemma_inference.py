"""Placeholder Gemma inference backend."""
from __future__ import annotations

from .base_inference import BaseInferenceModel


class GemmaInferenceModel(BaseInferenceModel):
    """Gemma model support is not implemented yet."""

    def load_model(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError(
            "GemmaInferenceModel is not implemented yet."
        )


__all__ = ["GemmaInferenceModel"]
