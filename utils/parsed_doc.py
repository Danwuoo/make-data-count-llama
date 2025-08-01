"""Unified schema for parsed documents."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .doi_recognizer import AccessionItem


class ParsedDoc(BaseModel):
    """Normalised representation of a parsed PDF or XML document."""

    doc_id: str
    source_type: Literal["pdf", "xml"]
    title: str
    abstract: str = ""
    body: str = ""
    references: List[str] = Field(default_factory=list)
    doi: Optional[str] = None
    accessions: List[AccessionItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = ["ParsedDoc"]
