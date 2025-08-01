"""L1: Parser utilities for processing raw PDF/XML inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import fitz

from .pdf_extractor import (
    LayoutValidator,
    PageSegmenter,
    ParagraphRestorer,
    PDFErrorCollector,
    ReferenceSplitter,
    SchemaEncoder,
    ParsedPDF,
    Section,
)
from .xml_extractor import XMLExtractor, ParsedXML
from .doi_recognizer import DOIRecognizer


class PDFExtractor:
    """High level interface for parsing PDFs into structured data."""

    def __init__(self) -> None:
        self.id_recognizer = DOIRecognizer()
        self.restorer = ParagraphRestorer()
        self.reference_splitter = ReferenceSplitter()
        self.validator = LayoutValidator()
        self.encoder = SchemaEncoder()
        self.errors = PDFErrorCollector()

    def extract(self, pdf_path: str) -> ParsedPDF:
        """Extract structured content from *pdf_path*."""
        try:
            document = fitz.open(pdf_path)
        except Exception as exc:
            self.errors.add("open_pdf", exc)
            raise

        segmenter = PageSegmenter(document)
        pages = segmenter.extract_pages()

        paragraphs_with_pages = []
        for page in pages:
            for para in self.restorer.restore(page["text"]):
                paragraphs_with_pages.append((para, page["page"]))

        full_text = "\n".join(p for p, _ in paragraphs_with_pages)
        ids = self.id_recognizer.recognize(full_text)
        doi = next((i.normalized for i in ids if i.id_type == "doi"), None)

        title_text = (
            paragraphs_with_pages[0][0]
            if paragraphs_with_pages
            else ""
        )
        abstract: Optional[str] = None
        body_start = 1
        if (
            len(paragraphs_with_pages) > 1
            and paragraphs_with_pages[1][0].lower().startswith("abstract")
        ):
            abstract = paragraphs_with_pages[1][0].split(":", 1)[-1].strip()
            body_start = 2

        body_paragraphs = paragraphs_with_pages[body_start:]
        references: list[Section] = []
        for idx, (text, page) in enumerate(body_paragraphs):
            if text.lower().startswith("references"):
                ref_text = "\n".join(t for t, _ in body_paragraphs[idx + 1:])
                references = [
                    Section(text=ref, pages=[])
                    for ref in self.reference_splitter.split(ref_text)
                ]
                body_paragraphs = body_paragraphs[:idx]
                break

        data = {
            "title": {
                "text": title_text,
                "pages": (
                    [paragraphs_with_pages[0][1]]
                    if paragraphs_with_pages
                    else []
                ),
            },
            "abstract": {
                "text": abstract,
                "pages": [paragraphs_with_pages[1][1]],
            }
            if abstract
            else None,
            "body": [
                {"text": text, "pages": [page]}
                for text, page in body_paragraphs
            ],
            "references": [ref.dict() for ref in references],
            "doi": doi,
        }

        parsed = self.encoder.encode(data)
        self.validator.validate(parsed)
        return parsed


ParsedDocument = Union[ParsedPDF, ParsedXML]


def parse_document(path: str) -> ParsedDocument:
    """Return a parsed representation for *path* regardless of format."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        extractor = PDFExtractor()
        return extractor.extract(path)
    extractor = XMLExtractor()
    return extractor.extract(path)
