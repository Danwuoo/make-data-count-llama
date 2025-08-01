"""Parser for Crossref XML documents."""

from __future__ import annotations

from lxml import etree


class CrossrefParser:
    """Extract minimal metadata from Crossref XML."""

    def _clean(self, text: str) -> str:
        return " ".join(text.split())

    def parse(self, root: etree._Element) -> dict:
        title_nodes = root.xpath('.//title')
        title_text = self._clean(" ".join(title_nodes[0].itertext())) if title_nodes else ""

        abstract_nodes = root.xpath('.//abstract')
        abstract_text = self._clean(" ".join(abstract_nodes[0].itertext())) if abstract_nodes else ""

        doi_nodes = root.xpath('.//doi')
        doi = doi_nodes[0].text.strip() if doi_nodes and doi_nodes[0].text else None

        data = {
            "title": {"text": title_text, "pages": []},
            "abstract": {"text": abstract_text, "pages": []} if abstract_text else None,
            "body": [],
            "references": [],
            "doi": doi,
        }
        return data
