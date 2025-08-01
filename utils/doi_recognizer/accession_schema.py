"""Schema definitions for accession matching results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class AccessionItem:
    """Structured output for accession identifiers."""

    raw_text: str
    standardized_id: str
    id_type: str
    score: float
    source: Dict[str, Optional[Any]]


__all__ = ["AccessionItem"]
