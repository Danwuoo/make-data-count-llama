"""Normalization helpers for identifiers."""

from __future__ import annotations

import re
import unicodedata


class DOINormalizer:
    """Normalize raw DOI strings."""

    def normalize(self, doi: str) -> str:
        doi = doi.strip()
        if doi.lower().startswith("doi:"):
            doi = doi.split(":", 1)[1]
        return f"doi:{doi.lower()}"


class AccessionNormalizer:
    """Normalize identifiers such as GEO, SRA or PDB codes."""

    PREFIXES = {
        "geo": "GSE",
        "sra": ("SRR", "SRP", "SRS", "SRX"),
        "ena": ("ERR", "ERP", "ERS", "ERX"),
        "ega": ("EGAD", "EGAS", "EGAE", "EGAN"),
        "pdb": "",
        "pmid": "PMID",
        "pmcid": "PMC",
    }

    def normalize(self, value: str, id_type: str) -> str:
        """Return a normalised identifier with standard prefix."""

        clean = unicodedata.normalize("NFKC", value)
        clean = re.sub(r"\s+", "", clean).upper()
        prefix = self.PREFIXES.get(id_type.lower(), "")

        if isinstance(prefix, tuple):
            if not any(clean.startswith(p) for p in prefix):
                clean = prefix[0] + re.sub(r"[^0-9A-Z]", "", clean)
        elif prefix and not clean.startswith(prefix):
            clean = prefix + re.sub(r"[^0-9A-Z]", "", clean)
        else:
            clean = re.sub(r"[^0-9A-Z]", "", clean)

        return f"{id_type.upper()}:{clean}"
