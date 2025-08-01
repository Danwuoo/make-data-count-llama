"""Basic validation for parsed XML documents."""

from __future__ import annotations

from .schema import ParsedXML


class LayoutValidator:
    """Perform minimal checks on parsed data."""

    def validate(self, parsed: ParsedXML) -> None:
        if not parsed.title.text:
            raise ValueError("title is required")
