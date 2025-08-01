from __future__ import annotations

from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    """Metadata describing the origin of a context unit."""

    section: str
    start_sentence_idx: int = Field(..., ge=0)
    end_sentence_idx: int = Field(..., ge=0)
    original_paragraph_id: int = Field(..., ge=0)


class ContextUnit(BaseModel):
    """Standard structure for a context unit passed to the LLM."""

    context_id: str
    doc_id: str
    text: str
    source: SourceInfo
    token_count: int
    importance_score: float = 0.0


__all__ = ["ContextUnit", "SourceInfo"]
