from __future__ import annotations

import re
from typing import List, Tuple

from ..parsed_doc import ParsedDoc
from .context_formatter import format_context
from .context_weighter import compute_importance
from .tokenizer_wrapper import TokenizerWrapper
from .validator import validate_contexts
from .schema import ContextUnit


SentenceInfo = Tuple[str, int, int, str]
# (section, paragraph_id, sentence_idx, text)


class SlidingWindowContext:
    """Build context units using a sliding window over document sentences."""

    def __init__(
        self,
        max_tokens: int = 512,
        stride: int = 128,
        tokenizer: TokenizerWrapper | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.stride = stride
        self.tokenizer = tokenizer or TokenizerWrapper()

    # ------------------------------------------------------------------
    # Sentence preparation
    # ------------------------------------------------------------------
    def _split_sentences(self, text: str) -> List[str]:
        pattern = r"(?<=[.!?])\s+"
        return [s.strip() for s in re.split(pattern, text) if s.strip()]

    def _prepare_sentences(self, doc: ParsedDoc) -> List[SentenceInfo]:
        sentences: List[SentenceInfo] = []
        # Title as single sentence
        if doc.title:
            sentences.append(("title", 0, 0, doc.title.strip()))
        # Abstract
        if doc.abstract:
            idx = 0
            for pid, para in enumerate(
                filter(None, [p.strip() for p in doc.abstract.split("\n")])
            ):
                for sent in self._split_sentences(para):
                    sentences.append(("abstract", pid, idx, sent))
                    idx += 1
        # Body
        if doc.body:
            idx = 0
            for pid, para in enumerate(
                filter(None, [p.strip() for p in doc.body.split("\n")])
            ):
                for sent in self._split_sentences(para):
                    sentences.append(("body", pid, idx, sent))
                    idx += 1
        return sentences

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------
    def _generate_windows(
        self, sentences: List[SentenceInfo]
    ) -> List[Tuple[int, int, int]]:
        token_counts = [
            self.tokenizer.count_tokens(s[3]) for s in sentences
        ]
        windows: List[Tuple[int, int, int]] = []
        start = 0
        n = len(sentences)
        while start < n:
            end = start
            total = 0
            while end < n and total + token_counts[end] <= self.max_tokens:
                total += token_counts[end]
                end += 1
            if start == end:
                end += 1
                total = token_counts[start]
            windows.append((start, end, total))
            if end >= n:
                break
            # compute new start to keep ``stride`` tokens overlap
            overlap = self.stride
            back = end
            kept = 0
            while back > start and kept < overlap:
                back -= 1
                kept += token_counts[back]
            start = back
        return windows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, doc: ParsedDoc) -> List[ContextUnit]:
        sentences = self._prepare_sentences(doc)
        total_sentences = len(sentences)
        windows = self._generate_windows(sentences)
        contexts: List[ContextUnit] = []
        for start, end, token_count in windows:
            snippet = sentences[start:end]
            text = " ".join(s[3] for s in snippet)
            section = snippet[0][0]
            start_sentence_idx = snippet[0][2]
            end_sentence_idx = snippet[-1][2]
            original_paragraph_id = snippet[0][1]
            importance = compute_importance(
                start_sentence_idx, total_sentences, section
            )
            ctx = format_context(
                doc_id=doc.doc_id,
                text=text,
                section=section,
                start_sentence_idx=start_sentence_idx,
                end_sentence_idx=end_sentence_idx,
                original_paragraph_id=original_paragraph_id,
                token_count=token_count,
                importance_score=importance,
                source_type=section,
            )
            contexts.append(ctx)
        return validate_contexts(contexts, self.max_tokens)


__all__ = ["SlidingWindowContext"]
