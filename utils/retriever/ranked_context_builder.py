"""Builders for :class:`~utils.retriever.schema.RankedContext`."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .schema import RankedContext


class RankedContextBuilder:
    """Convenience helper to instantiate :class:`RankedContext`."""

    def build(
        self,
        *,
        query_text: str,
        context: str,
        rank: int,
        similarity_score: float,
        final_score: float,
        reranker_score: Optional[float] = None,
        doc_id: Optional[str] = None,
        context_id: Optional[str] = None,
        label: Optional[str] = None,
        section: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankedContext:
        return RankedContext(
            query_text=query_text,
            context=context,
            rank=rank,
            similarity_score=similarity_score,
            final_score=final_score,
            reranker_score=reranker_score,
            doc_id=doc_id,
            context_id=context_id,
            label=label,
            section=section,
            metadata=metadata or {},
        )
