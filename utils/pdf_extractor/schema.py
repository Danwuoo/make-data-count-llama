"""Pydantic models for parsed PDF structures."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


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


class SchemaEncoder:
    """Convert dictionaries into :class:`ParsedPDF` objects."""

    def encode(self, data: dict) -> ParsedPDF:
        return ParsedPDF(**data)
