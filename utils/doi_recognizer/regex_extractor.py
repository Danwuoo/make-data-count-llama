"""Regular expression based ID extractor."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class RegexMatch:
    """A raw regex match for an identifier."""

    id_type: str
    value: str
    start: int
    end: int


class RegexExtractor:
    """Extract accession-like identifiers using regex patterns."""

    PATTERNS = {
        "doi": re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I),
        "pmid": re.compile(r"PMID[:\s]*\d+", re.I),
        "pmcid": re.compile(r"PMC(?:ID)?[:\s]*\d+", re.I),
        "gse": re.compile(r"GSE\d{3,6}", re.I),
        "srr": re.compile(r"SRR\d{3,6}", re.I),
        "pdb": re.compile(r"PDB[\s-]?ID[:\s]*[0-9A-Z]{4}", re.I),
    }

    def extract(self, text: str) -> List[RegexMatch]:
        matches: List[RegexMatch] = []
        for id_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                matches.append(
                    RegexMatch(
                        id_type=id_type,
                        value=match.group(0),
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return matches
