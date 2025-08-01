"""Context builder package providing sliding window functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

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


def build_from_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    max_tokens: int = 512,
    stride: int = 128,
) -> None:
    """Read ``ParsedDoc`` objects from ``input_path`` and write context units."""

    path_in = Path(input_path)
    path_out = Path(output_path)

    def _iter_docs() -> Iterable[ParsedDoc]:
        with path_in.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield ParsedDoc.model_validate_json(line)

    contexts: List[ContextUnit] = []
    for doc in _iter_docs():
        contexts.extend(
            build_context(doc, max_tokens=max_tokens, stride=stride)
        )

    with path_out.open("w", encoding="utf-8") as f:
        for ctx in contexts:
            f.write(ctx.model_dump_json() + "\n")


__all__ = [
    "build_context",
    "build_from_jsonl",
    "ContextUnit",
    "SlidingWindowContext",
    "TitleAbstractMerger",
]
