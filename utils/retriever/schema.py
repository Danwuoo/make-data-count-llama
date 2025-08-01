"""Pydantic schemas for retrieval results."""

from __future__ import annotations

from pydantic import BaseModel


class RetrievalResultItem(BaseModel):
    query_text: str
    matched_context: str
    similarity_score: float
    doc_id: str | None = None
    context_id: str | None = None
    section: str | None = None
