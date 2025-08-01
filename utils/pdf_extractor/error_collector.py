"""Collect errors encountered during PDF extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PDFError:
    """Single extraction error entry."""

    message: str
    exception: Optional[str] = None


@dataclass
class PDFErrorCollector:
    """Store multiple :class:`PDFError` instances."""

    errors: List[PDFError] = field(default_factory=list)

    def add(self, message: str, exc: Optional[Exception] = None) -> None:
        self.errors.append(PDFError(message, repr(exc) if exc else None))
