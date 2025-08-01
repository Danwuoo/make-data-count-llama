from __future__ import annotations

import uuid

from .schema import ContextUnit, SourceInfo


def format_context(
    doc_id: str,
    text: str,
    section: str,
    start_sentence_idx: int,
    end_sentence_idx: int,
    original_paragraph_id: int,
    token_count: int,
    importance_score: float,
    source_type: str | None = None,
) -> ContextUnit:
    """Create a :class:`ContextUnit` with standard metadata."""

    context_id = f"ctx_{uuid.uuid4().hex[:8]}"
    source = SourceInfo(
        section=section,
        start_sentence_idx=start_sentence_idx,
        end_sentence_idx=end_sentence_idx,
        original_paragraph_id=original_paragraph_id,
        source_type=source_type or section,
    )
    return ContextUnit(
        context_id=context_id,
        doc_id=doc_id,
        text=text,
        source=source,
        token_count=token_count,
        importance_score=importance_score,
    )


__all__ = ["format_context"]
