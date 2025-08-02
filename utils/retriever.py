"""Convenience helpers for semantic context retrieval."""

from __future__ import annotations

from pathlib import Path

from .retriever.context_retriever import ContextRetriever
from .retriever.embedding_encoder import EmbeddingEncoder
from .retriever.index_storage import IndexStorageManager

__all__ = ["ContextRetriever", "retrieve"]


def retrieve(
    query: str,
    *,
    index_path: str | Path,
    metadata_path: str | Path,
    top_k: int = 5,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> list[str]:
    """Return context texts ranked by similarity to ``query``.

    This helper loads a :class:`ContextRetriever` from persisted index files
    created via :class:`~utils.retriever.memory_builder.MemoryBuilder` and
    queries it for the most relevant contexts.

    Parameters
    ----------
    query:
        Search query to encode and match against the index.
    index_path, metadata_path:
        Locations of the FAISS index and accompanying metadata JSON.
    top_k:
        Maximum number of results to return. Defaults to 5.
    model_name:
        SentenceTransformer model used for encoding when no custom model is
        supplied. Defaults to ``"sentence-transformers/all-mpnet-base-v2"``.

    Returns
    -------
    list[str]
        Ranked context texts most relevant to ``query``.
    """

    encoder = EmbeddingEncoder(model_name=model_name)
    storage = IndexStorageManager(index_path=index_path, metadata_path=metadata_path)
    retriever = ContextRetriever.from_storage(encoder=encoder, storage=storage)
    results = retriever.retrieve(query, top_k=top_k)
    return [item.matched_context for item in results]

