"""L1: Parser utilities for processing raw PDF/XML inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from dataclasses import asdict

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
from .doi_recognizer import DOIRecognizer, AccessionMatcher
from .parsed_doc import ParsedDoc


class PDFExtractor:
    """High level interface for parsing PDFs into structured data."""

    def __init__(self) -> None:
        self.id_recognizer = DOIRecognizer()
        self.restorer = ParagraphRestorer()
        self.reference_splitter = ReferenceSplitter()
        self.validator = LayoutValidator()
        self.encoder = SchemaEncoder()
        self.errors = PDFErrorCollector()
        self.acc_matcher = AccessionMatcher()

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
        accessions = self.acc_matcher.match(full_text)

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
            "accessions": [asdict(a) for a in accessions],
        }

        parsed = self.encoder.encode(data)
        self.validator.validate(parsed)
        return parsed


class FileParser:
    """Dispatch to the appropriate extractor and normalise output."""

    def parse(self, path: str) -> ParsedDoc:
        ext = Path(path).suffix.lower()
        doc_id = Path(path).stem
        if ext == ".pdf":
            extractor = PDFExtractor()
            parsed: ParsedPDF = extractor.extract(path)
            with fitz.open(path) as doc:
                page_count = doc.page_count
            return ParsedDoc(
                doc_id=doc_id,
                source_type="pdf",
                title=parsed.title.text,
                abstract=parsed.abstract.text if parsed.abstract else "",
                body="\n".join(sec.text for sec in parsed.body),
                references=[sec.text for sec in parsed.references],
                doi=parsed.doi,
                accessions=parsed.accessions,
                metadata={"page_count": page_count},
            )
        extractor = XMLExtractor()
        parsed_xml: ParsedXML = extractor.extract(path)
        return ParsedDoc(
            doc_id=doc_id,
            source_type="xml",
            title=parsed_xml.title.text,
            abstract=parsed_xml.abstract.text if parsed_xml.abstract else "",
            body="\n".join(sec.text for sec in parsed_xml.body),
            references=[sec.text for sec in parsed_xml.references],
            doi=parsed_xml.doi,
            accessions=parsed_xml.accessions,
            metadata={},
        )


def parse_document(path: str) -> ParsedDoc:
    """Return a :class:`ParsedDoc` representation for *path*."""
    parser = FileParser()
    return parser.parse(path)
