"""Detect the format of XML documents."""

from __future__ import annotations

from lxml import etree


class XMLFormatDetector:
    """Classify XML documents by root tag and namespace."""

    def detect(self, root: etree._Element) -> str:
        tag = etree.QName(root).localname.lower()
        if tag == "collection":
            return "bioc"
        if tag == "doi_batch":
            return "crossref"
        if tag == "article":
            return "jats"
        return "unknown"
