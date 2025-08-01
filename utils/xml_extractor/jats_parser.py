"""Parser for JATS XML documents."""

from __future__ import annotations

from lxml import etree

from .reference_builder import ReferenceBuilder
from .schema import Section


class JATSParser:
    """Extract structured content from JATS articles."""

    def __init__(self) -> None:
        self.references = ReferenceBuilder()

    def _clean(self, text: str) -> str:
        return " ".join(text.split())

    def parse(self, root: etree._Element) -> dict:
        def text_or_empty(xpath: str) -> str:
            nodes = root.xpath(xpath)
            if not nodes:
                return ""
            text = " ".join(" ".join(n.itertext()) for n in nodes)
            return self._clean(text)

        title_text = text_or_empty('.//article-title')
        if not title_text:
            title_text = text_or_empty('.//title-group/article-title')

        abstract_text = text_or_empty('.//abstract')

        body_paras = [
            Section(text=self._clean(" ".join(p.itertext())), pages=[])
            for p in root.xpath('.//body//p')
        ]

        ref_texts = self.references.build(root.xpath('.//ref-list//ref'))
        refs = [Section(text=txt, pages=[]) for txt in ref_texts]

        doi_nodes = root.xpath(".//article-id[@pub-id-type='doi']/text()")
        doi = doi_nodes[0].strip() if doi_nodes else None

        data = {
            "title": {"text": title_text, "pages": []},
            "abstract": {"text": abstract_text, "pages": []} if abstract_text else None,
            "body": [s.dict() for s in body_paras],
            "references": [s.dict() for s in refs],
            "doi": doi,
        }
        return data
