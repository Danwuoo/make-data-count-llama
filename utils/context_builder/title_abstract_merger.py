from __future__ import annotations

import regex as re

from ..parsed_doc import ParsedDoc
from .context_unit_builder import ContextUnitBuilder
from .text_formatter import TextFormatter
from .tokenizer_wrapper import TokenizerWrapper
from .schema import ContextUnit


class TitleAbstractMerger:
    """Merge a document's title and abstract into a single context unit."""

    def __init__(
        self,
        formatter: TextFormatter | None = None,
        tokenizer: TokenizerWrapper | None = None,
        builder: ContextUnitBuilder | None = None,
    ) -> None:
        self.formatter = formatter or TextFormatter()
        self.tokenizer = tokenizer or TokenizerWrapper()
        self.builder = builder or ContextUnitBuilder()

    # ------------------------------------------------------------------
    def _fallback_abstract(self, body: str) -> str:
        """Use the first couple of sentences from the body when abstract is
        missing."""
        if not body:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        return " ".join(sentences[:2]).strip()

    # ------------------------------------------------------------------
    def merge(self, doc: ParsedDoc) -> ContextUnit:
        abstract = doc.abstract or self._fallback_abstract(doc.body)
        text = self.formatter.format(doc.title, abstract)
        token_count = self.tokenizer.count_tokens(text)
        return self.builder.build(
            doc_id=doc.doc_id,
            text=text,
            section="intro",
            token_count=token_count,
            importance_score=1.0,
            source_type="title+abstract",
        )


__all__ = ["TitleAbstractMerger"]
