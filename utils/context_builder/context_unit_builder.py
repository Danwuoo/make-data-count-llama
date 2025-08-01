from __future__ import annotations

import uuid

from .schema import ContextUnit, SourceInfo


class ContextUnitBuilder:
    """Construct :class:`ContextUnit` objects with standard metadata."""

    def build(
        self,
        doc_id: str,
        text: str,
        section: str,
        token_count: int,
        importance_score: float,
        source_type: str | None = None,
    ) -> ContextUnit:
        context_id = f"ctx_{uuid.uuid4().hex[:8]}"
        source = SourceInfo(
            section=section,
            start_sentence_idx=0,
            end_sentence_idx=0,
            original_paragraph_id=0,
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


__all__ = ["ContextUnitBuilder"]
