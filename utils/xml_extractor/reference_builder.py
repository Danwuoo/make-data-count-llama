"""Build reference strings from XML reference elements."""

from __future__ import annotations

from lxml import etree


class ReferenceBuilder:
    """Combine nested XML reference tags into strings."""

    def build(self, ref_elements: list[etree._Element]) -> list[str]:
        refs: list[str] = []
        for ref in ref_elements:
            text = " ".join(ref.itertext())
            text = " ".join(text.split())
            if text:
                refs.append(text)
        return refs
