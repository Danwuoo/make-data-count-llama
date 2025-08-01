"""Context builder package providing sliding window functionality."""

from __future__ import annotations

from typing import List

from ..parsed_doc import ParsedDoc
from .schema import ContextUnit
from .sliding_window import SlidingWindowContext


def build_context(
    parsed_doc: ParsedDoc, max_tokens: int = 512, stride: int = 128
) -> List[ContextUnit]:
    """Build context units from a parsed document using a sliding window."""
    builder = SlidingWindowContext(max_tokens=max_tokens, stride=stride)
    return builder.build(parsed_doc)


__all__ = ["build_context", "ContextUnit", "SlidingWindowContext"]
