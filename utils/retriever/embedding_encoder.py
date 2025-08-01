"""Text embedding utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class EmbeddingEncoder:
    """Encode text into dense vectors using SentenceTransformers.

    Parameters
    ----------
    model_name: str, optional
        Name of the model to load from ``sentence-transformers``.
        Defaults to ``"sentence-transformers/all-mpnet-base-v2"``.
    model: object, optional
        Preloaded model implementing an ``encode`` method. Primarily useful
        for testing where a lightweight stub can be supplied.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        model: object | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: Sequence[str] | str) -> np.ndarray:
        """Return embeddings for ``texts``.

        Parameters
        ----------
        texts:
            A single string or a sequence of strings to encode.
        """

        self._load_model()
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(list(texts))
        return np.asarray(embeddings, dtype="float32")
