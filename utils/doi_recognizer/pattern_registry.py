"""Regular expression patterns for accession identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PatternMatch:
    """Raw regex match for an accession identifier."""

    id_type: str
    value: str
    start: int
    end: int


class PatternRegistry:
    """Registry holding regex patterns for common accession identifiers."""

    PATTERNS: Dict[str, re.Pattern] = {
        "GEO": re.compile(r"GSE\d{3,7}", re.I),
        "SRA": re.compile(r"S(?:RR|RP|RS|RX)\d{5,9}", re.I),
        "PDB": re.compile(r"(?:PDB[:\s-]*)?[0-9A-Z]{4}", re.I),
        "EGA": re.compile(r"EGA[DS]\d{11,13}", re.I),
        "ENA": re.compile(r"ER[RXDS]\d{6,10}", re.I),
    }

    def find(self, text: str) -> List[PatternMatch]:
        matches: List[PatternMatch] = []
        for id_type, pattern in self.PATTERNS.items():
            for m in pattern.finditer(text):
                matches.append(
                    PatternMatch(
                        id_type=id_type,
                        value=m.group(0),
                        start=m.start(),
                        end=m.end(),
                    )
                )
        return matches


__all__ = ["PatternRegistry", "PatternMatch"]
