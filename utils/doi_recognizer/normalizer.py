"""Normalization helpers for identifiers."""

from __future__ import annotations

import re


class DOINormalizer:
    """Normalize raw DOI strings."""

    def normalize(self, doi: str) -> str:
        doi = doi.strip()
        if doi.lower().startswith("doi:"):
            doi = doi.split(":", 1)[1]
        return f"doi:{doi.lower()}"


class AccessionNormalizer:
    """Normalize other identifier types such as PMID and GSE."""

    def normalize(self, value: str, id_type: str) -> str:
        clean = re.sub(r"[^0-9A-Za-z]", "", value)
        if id_type == "pmcid" and not clean.upper().startswith("PMC"):
            clean = f"PMC{clean}"
        return f"{id_type}:{clean.lower()}"
