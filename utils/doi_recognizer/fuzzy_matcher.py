"""Fuzzy matching helpers for broken identifiers."""

from __future__ import annotations

import re
from typing import List

from .regex_extractor import RegexMatch


class FuzzyMatcher:
    """Attempt to recover identifiers split by whitespace or newlines."""

    DOI_FUZZY = re.compile(
        r"10\s*[.](?:\s*\d){4,9}\s*/\s*[-._;()/:A-Z0-9]+",
        re.I | re.DOTALL,
    )

    def extract(self, text: str) -> List[RegexMatch]:
        matches: List[RegexMatch] = []
        for match in self.DOI_FUZZY.finditer(text):
            cleaned = re.sub(r"\s+", "", match.group(0))
            matches.append(
                RegexMatch("doi", cleaned, match.start(), match.end())
            )
        return matches
