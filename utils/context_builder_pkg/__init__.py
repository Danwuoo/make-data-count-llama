"""Context builder package providing sliding window functionality."""

from __future__ import annotations

from typing import List

from ..parsed_doc import ParsedDoc
from .schema import ContextUnit
from .sliding_window import SlidingWindowContext
from .title_abstract_merger import TitleAbstractMerger


def build_context(
    parsed_doc: ParsedDoc, max_tokens: int = 512, stride: int = 128
) -> List[ContextUnit]:
    """Build context units from a parsed document."""
    merger = TitleAbstractMerger()
    merged = merger.merge(parsed_doc)
    builder = SlidingWindowContext(max_tokens=max_tokens, stride=stride)
    windows = builder.build(parsed_doc)
    return [merged, *windows]


__all__ = [
    "build_context",
    "ContextUnit",
    "SlidingWindowContext",
    "TitleAbstractMerger",
]
