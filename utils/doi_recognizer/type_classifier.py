"""Fallback identifier type classification."""

from __future__ import annotations

from typing import Optional


class IDTypeClassifier:
    """Guess identifier types based on simple heuristics."""

    def classify(self, value: str) -> Optional[str]:
        value_upper = value.upper()
        if value_upper.startswith("10."):
            return "doi"
        if value_upper.startswith("PMCID") or value_upper.startswith("PMC"):
            return "pmcid"
        if value_upper.startswith("PMID"):
            return "pmid"
        if value_upper.startswith("GSE"):
            return "gse"
        if value_upper.startswith("SRR"):
            return "srr"
        if value_upper.startswith("PDB"):
            return "pdb"
        return None
