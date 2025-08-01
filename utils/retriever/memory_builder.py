"""Utility to build and persist vector-based context memory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict

from .embedding_encoder import EmbeddingEncoder
from .vector_indexer import VectorIndexer
from .index_storage import IndexStorageManager


class MemoryBuilder:
    """Build a FAISS index from context units and save with metadata.

    Parameters
    ----------
    encoder:
        Optional pre-initialised :class:`EmbeddingEncoder`. If not provided a
        default encoder instance will be created lazily when building the
        memory.
    metric:
        Distance metric used by :class:`VectorIndexer`.
        Defaults to ``"cosine"``.
    """

    def __init__(
        self,
        *,
        encoder: EmbeddingEncoder | None = None,
        metric: str = "cosine",
    ) -> None:
        self.encoder = encoder or EmbeddingEncoder()
        self.metric = metric

    def build(
        self, contexts: Iterable[Dict[str, str]]
    ) -> tuple[VectorIndexer, List[dict]]:
        """Return an indexer and metadata list built from ``contexts``."""

        context_list = list(contexts)
        texts = [c.get("text", "") for c in context_list]
        vectors = self.encoder.encode(texts)
        indexer = VectorIndexer(
            dimension=vectors.shape[1], metric=self.metric
        )
        indexer.add(vectors)
        return indexer, context_list

    def build_from_jsonl(
        self, path: str | Path
    ) -> tuple[VectorIndexer, List[dict]]:
        """Load contexts from ``path`` (JSONL) and build memory."""

        with open(path, "r", encoding="utf-8") as f:
            contexts = [json.loads(line) for line in f if line.strip()]
        return self.build(contexts)

    def build_and_save(
        self,
        contexts: Iterable[Dict[str, str]],
        storage: IndexStorageManager,
    ) -> None:
        """Build memory from ``contexts`` and persist via ``storage``."""

        indexer, metadata = self.build(contexts)
        storage.save(indexer, metadata)
