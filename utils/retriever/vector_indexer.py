"""FAISS vector index abstraction."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class VectorIndexer:
    """FAISS index wrapper for cosine or inner product search."""

    def __init__(self, dimension: int, metric: str = "cosine") -> None:
        self.dimension = dimension
        self.metric = metric
        if metric not in {"cosine", "ip"}:
            raise ValueError(
                "metric must be 'cosine' or 'ip'"
            )
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("vectors shape mismatch")
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(
        self, query: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if query.ndim == 1:
            query = query[None, :]
        if self.metric == "cosine":
            faiss.normalize_L2(query)
        scores, ids = self.index.search(query, top_k)
        return ids[0], scores[0]

    def save(self, path: str | Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path, metric: str = "cosine") -> "VectorIndexer":
        index = faiss.read_index(str(path))
        obj = cls(index.d, metric=metric)
        obj.index = index
        return obj
