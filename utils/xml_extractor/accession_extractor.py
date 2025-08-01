"""Extract accession identifiers from text."""

from __future__ import annotations

import re
from typing import List


class AccessionExtractor:
    """Find common accession identifiers such as GSE, PMID, and PDB."""

    GSE = re.compile(r"GSE\d+", re.IGNORECASE)
    PMID = re.compile(r"PMID\d+", re.IGNORECASE)
    PDB = re.compile(r"PDB\w+", re.IGNORECASE)

    def extract(self, text: str) -> List[str]:
        ids = set()
        for pattern in (self.GSE, self.PMID, self.PDB):
            ids.update(match.group(0).upper() for match in pattern.finditer(text))
        return sorted(ids)
