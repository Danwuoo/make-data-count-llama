from __future__ import annotations

import textwrap
import regex as re
from ftfy import fix_text


class TextFormatter:
    """Utility for cleaning and tagging title and abstract text."""

    title_tag = "[TITLE]:"
    abstract_tag = "[ABSTRACT]:"

    def _clean(self, text: str) -> str:
        cleaned = fix_text(text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def format(self, title: str, abstract: str) -> str:
        title_clean = self._clean(title)
        abstract_clean = self._clean(abstract)
        merged = (
            f"{self.title_tag} {title_clean}\n"
            f"{self.abstract_tag} {abstract_clean}"
        ).strip()
        return textwrap.dedent(merged).strip()


__all__ = ["TextFormatter"]
