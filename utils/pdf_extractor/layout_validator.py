"""Validate the minimal structure of an extracted PDF."""

from __future__ import annotations

from .schema import ParsedPDF


class LayoutValidator:
    """Ensure required fields are present."""

    def validate(self, parsed: ParsedPDF) -> None:
        if not parsed.title.text.strip():
            raise ValueError("Title is missing")
        if not parsed.body:
            raise ValueError("Body is missing")
