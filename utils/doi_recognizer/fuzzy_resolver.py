"""Fuzzy matching for imperfect accession identifiers."""

from __future__ import annotations

import re
import unicodedata
from typing import List

from fuzzywuzzy import fuzz

from .pattern_registry import PatternMatch


class FuzzyAccessionResolver:
    """Resolve malformed or partial accession identifiers using fuzzy logic."""

    def __init__(self) -> None:
        self.prefixes = {
            "GEO": "GSE",
            "SRA": "SRR",
            "EGA": "EGAD",
            "ENA": "ER",
        }
        self.candidate = re.compile(
            r"([A-Za-z]{2,5})[\s-]*([0-9]{3,13})",
            re.I,
        )

    def resolve(self, text: str) -> List[PatternMatch]:
        norm = unicodedata.normalize("NFKC", text)
        matches: List[PatternMatch] = []
        for m in self.candidate.finditer(norm):
            prefix = m.group(1).upper()
            digits = m.group(2)
            for id_type, expected in self.prefixes.items():
                if fuzz.ratio(prefix, expected) >= 80:
                    value = expected + digits
                    matches.append(
                        PatternMatch(
                            id_type=id_type,
                            value=value,
                            start=m.start(),
                            end=m.end(),
                        )
                    )
                    break
        return matches


__all__ = ["FuzzyAccessionResolver"]
