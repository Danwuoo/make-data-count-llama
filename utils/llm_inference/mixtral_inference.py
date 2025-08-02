"""Placeholder Mixtral inference backend."""
from __future__ import annotations

from .base_inference import BaseInferenceModel


class MixtralInferenceModel(BaseInferenceModel):
    """Mixtral model support is not implemented yet."""

    def load_model(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError(
            "MixtralInferenceModel is not implemented yet."
        )


__all__ = ["MixtralInferenceModel"]
