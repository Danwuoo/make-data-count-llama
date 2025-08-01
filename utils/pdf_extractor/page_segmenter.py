"""Utilities for segmenting a PDF document into page-level text blocks."""

from __future__ import annotations

from typing import List, Dict

import fitz


class PageSegmenter:
    """Extract text from each page of a PDF."""

    def __init__(self, document: fitz.Document) -> None:
        self.document = document

    def extract_pages(self) -> List[Dict[str, object]]:
        """Return a list with page number and text content."""
        pages: List[Dict[str, object]] = []
        for number, page in enumerate(self.document, start=1):
            text = page.get_text("text")
            pages.append({"page": number, "text": text})
        return pages
