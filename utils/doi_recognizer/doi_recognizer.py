"""High level identifier recognition utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .regex_extractor import RegexExtractor
from .fuzzy_matcher import FuzzyMatcher
from .normalizer import DOINormalizer, AccessionNormalizer
from .type_classifier import IDTypeClassifier
from .semantic_locator import SemanticZoneLocator
from .remote_validator import RemoteValidator


@dataclass
class StructuredID:
    """A structured representation of an identifier."""

    id_type: str
    raw: str
    normalized: str
    page: Optional[int] = None
    section: Optional[str] = None


class DOIRecognizer:
    """Recognize and normalise various document identifiers."""

    def __init__(self) -> None:
        self.regex = RegexExtractor()
        self.fuzzy = FuzzyMatcher()
        self.doi_norm = DOINormalizer()
        self.acc_norm = AccessionNormalizer()
        self.classifier = IDTypeClassifier()
        self.locator = SemanticZoneLocator()
        self.validator = RemoteValidator()

    def recognize(self, text: str, meta: Optional[dict] = None) -> List[StructuredID]:
        """Return structured identifiers found in *text*."""

        meta = meta or {}
        matches = self.regex.extract(text)
        if not matches:
            matches = self.fuzzy.extract(text)

        ids: List[StructuredID] = []
        for match in matches:
            normalized = (
                self.doi_norm.normalize(match.value)
                if match.id_type == "doi"
                else self.acc_norm.normalize(match.value, match.id_type)
            )
            ids.append(
                StructuredID(
                    id_type=match.id_type,
                    raw=match.value,
                    normalized=normalized,
                    page=meta.get("page"),
                    section=meta.get("section"),
                )
            )

        ids = self.locator.filter(ids, meta)
        return [i for i in ids if self.validator.validate(i.normalized, i.id_type)]
