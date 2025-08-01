"""Pydantic schemas for retrieval and ranking results."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class RetrievalResultItem(BaseModel):
    """Structure returned by :class:`ContextRetriever`."""

    query_text: str
    matched_context: str
    similarity_score: float
    doc_id: Optional[str] = None
    context_id: Optional[str] = None
    section: Optional[str] = None


class RankedContext(BaseModel):
    """Structure representing a ranked context segment."""

    query_text: str
    rank: int
    similarity_score: float
    final_score: float
    context: str
    reranker_score: Optional[float] = None
    doc_id: Optional[str] = None
    context_id: Optional[str] = None
    label: Optional[str] = None
    section: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

