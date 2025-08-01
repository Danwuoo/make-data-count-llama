from __future__ import annotations

"""High level context builder for turning parsed docs into context units."""

from pathlib import Path
from typing import Iterable, List

from .parsed_doc import ParsedDoc
from .context_builder_pkg import (
    ContextUnit,
    SlidingWindowContext,
    TitleAbstractMerger,
)


def build_context(
    parsed_doc: ParsedDoc, max_tokens: int = 512, stride: int = 128
) -> List[ContextUnit]:
    """Build context units for a single :class:`ParsedDoc`.

    Parameters
    ----------
    parsed_doc:
        The parsed document from L1.
    max_tokens:
        Maximum number of tokens allowed per context window.
    stride:
        Overlap in tokens between consecutive windows.
    """
    merger = TitleAbstractMerger()
    intro_ctx = merger.merge(parsed_doc)
    window_builder = SlidingWindowContext(max_tokens=max_tokens, stride=stride)
    windows = window_builder.build(parsed_doc)
    return [intro_ctx, *windows]


def build_from_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    max_tokens: int = 512,
    stride: int = 128,
) -> None:
    """Read ``ParsedDoc`` objects from ``input_path`` and write context units.

    The output is a JSON lines file where each line represents a
    :class:`ContextUnit` in its JSON serialised form.
    """

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


__all__ = ["build_context", "build_from_jsonl", "ContextUnit"]
