"""High level interface for semantic context retrieval."""

from __future__ import annotations

from typing import Dict, List

from .context_filter import ContextFilter
from .embedding_encoder import EmbeddingEncoder
from .index_storage import IndexStorageManager
from .schema import RetrievalResultItem
from .vector_indexer import VectorIndexer


class ContextRetriever:
    """Retrieve relevant context units from a vector index."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        indexer: VectorIndexer,
        metadata: List[dict],
        context_filter: ContextFilter | None = None,
    ) -> None:
        self.encoder = encoder
        self.indexer = indexer
        self.metadata = metadata
        self.context_filter = context_filter or ContextFilter()

    @classmethod
    def from_storage(
        cls,
        encoder: EmbeddingEncoder,
        storage: IndexStorageManager,
    ) -> "ContextRetriever":
        indexer, metadata = storage.load()
        return cls(encoder=encoder, indexer=indexer, metadata=metadata)

    def retrieve(
        self, query: str, top_k: int = 5, filters: Dict[str, str] | None = None
    ) -> List[RetrievalResultItem]:
        query_vec = self.encoder.encode(query)
        ids, scores = self.indexer.search(query_vec, top_k * 5)
        results: List[RetrievalResultItem] = []
        for idx, score in zip(ids, scores):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if filters and not self.context_filter.match(meta, filters):
                continue
            results.append(
                RetrievalResultItem(
                    query_text=query,
                    matched_context=meta.get("text", ""),
                    similarity_score=float(score),
                    doc_id=meta.get("doc_id"),
                    context_id=meta.get("context_id"),
                    section=meta.get("section"),
                )
            )
            if len(results) >= top_k:
                break
        return results
