"""Pydantic models for parsed XML structures."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Section(BaseModel):
    """A text block and the source it originates from."""

    text: str
    pages: List[int] = Field(default_factory=list)


class ParsedXML(BaseModel):
    """Representation of a parsed XML document."""

    title: Section
    abstract: Optional[Section] = None
    body: List[Section] = Field(default_factory=list)
    references: List[Section] = Field(default_factory=list)
    doi: Optional[str] = None
    accessions: List[str] = Field(default_factory=list)


class SchemaEncoder:
    """Convert dictionaries into :class:`ParsedXML` objects."""

    def encode(self, data: dict) -> ParsedXML:
        return ParsedXML(**data)
