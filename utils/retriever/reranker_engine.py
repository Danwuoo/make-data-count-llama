"""Interface for optional cross-encoder reranking models."""

from __future__ import annotations

from typing import List, Tuple

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None  # type: ignore


class RerankerEngine:
    """Lightweight wrapper around a cross-encoder model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model = CrossEncoder(model_name) if model_name and CrossEncoder else None

    def score(self, query: str, context: str) -> float:
        """Return a reranker score for ``(query, context)``.

        If no model is available, ``0.0`` is returned.
        """

        if not self.model:
            return 0.0
        pair: List[Tuple[str, str]] = [(query, context)]
        return float(self.model.predict(pair)[0])
