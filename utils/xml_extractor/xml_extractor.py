"""High level interface for parsing XML documents."""

from __future__ import annotations

from typing import Dict

from lxml import etree

from ..doi_recognizer import AccessionMatcher
from .crossref_parser import CrossrefParser
from .error_collector import XMLErrorCollector
from .format_detector import XMLFormatDetector
from .jats_parser import JATSParser
from .layout_validator import LayoutValidator
from .schema import ParsedXML, SchemaEncoder


class XMLExtractor:
    """Parse XML documents into structured data."""

    def __init__(self) -> None:
        self.detector = XMLFormatDetector()
        self.jats = JATSParser()
        self.crossref = CrossrefParser()
        self.accessions = AccessionMatcher()
        self.validator = LayoutValidator()
        self.encoder = SchemaEncoder()
        self.errors = XMLErrorCollector()

    def extract(self, xml_path: str) -> ParsedXML:
        try:
            tree = etree.parse(xml_path)
            root = tree.getroot()
        except Exception as exc:
            self.errors.add("parse_xml", exc)
            raise

        fmt = self.detector.detect(root)
        parser_map: Dict[str, object] = {
            "jats": self.jats,
            "crossref": self.crossref,
        }
        parser = parser_map.get(fmt, self.jats)
        try:
            data = parser.parse(root)
        except Exception as exc:
            self.errors.add("parse_content", exc)
            data = {
                "title": {"text": "", "pages": []},
                "abstract": None,
                "body": [],
                "references": [],
                "doi": None,
            }

        text_blocks = [data["title"]["text"]]
        if data.get("abstract"):
            text_blocks.append(data["abstract"]["text"])
        text_blocks.extend(section["text"] for section in data.get("body", []))
        text_blocks.extend(
            section["text"] for section in data.get("references", [])
        )
        all_text = " ".join(text_blocks)
        data["accessions"] = [
            a.__dict__ for a in self.accessions.match(all_text)
        ]

        parsed = self.encoder.encode(data)
        self.validator.validate(parsed)
        return parsed
