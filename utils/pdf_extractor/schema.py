"""Pydantic models for parsed PDF structures."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from ..doi_recognizer import AccessionItem


class Section(BaseModel):
    """A text block and the pages it originates from."""

    text: str
    pages: List[int] = Field(default_factory=list)


class ParsedPDF(BaseModel):
    """Representation of a parsed PDF document."""

    title: Section
    abstract: Optional[Section] = None
    body: List[Section] = Field(default_factory=list)
    references: List[Section] = Field(default_factory=list)
    doi: Optional[str] = None
    accessions: List[AccessionItem] = Field(default_factory=list)


class SchemaEncoder:
    """Convert dictionaries into :class:`ParsedPDF` objects."""

    def encode(self, data: dict) -> ParsedPDF:
        return ParsedPDF(**data)
