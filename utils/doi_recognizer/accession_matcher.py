"""High level matcher that extracts accession identifiers from text."""

from __future__ import annotations

from typing import List, Optional

from .pattern_registry import PatternRegistry, PatternMatch
from .fuzzy_resolver import FuzzyAccessionResolver
from .context_weighter import ContextWeighter
from .normalizer import AccessionNormalizer
from .accession_schema import AccessionItem


class AccessionMatcher:
    """Extract and normalise common accession identifiers."""

    def __init__(self) -> None:
        self.registry = PatternRegistry()
        self.fuzzy = FuzzyAccessionResolver()
        self.context = ContextWeighter()
        self.normalizer = AccessionNormalizer()

    def match(
        self, text: str, meta: Optional[dict] = None
    ) -> List[AccessionItem]:
        meta = meta or {}
        raw_matches: List[PatternMatch] = self.registry.find(text)
        raw_matches.extend(self.fuzzy.resolve(text))

        items: List[AccessionItem] = []
        seen = set()
        for m in raw_matches:
            key = (m.id_type, m.start, m.end)
            if key in seen:
                continue
            seen.add(key)
            score = self.context.weight(text, m, meta)
            standardized = self.normalizer.normalize(m.value, m.id_type)
            items.append(
                AccessionItem(
                    raw_text=m.value,
                    standardized_id=standardized,
                    id_type=m.id_type,
                    score=score,
                    source={
                        "page": meta.get("page"),
                        "section": meta.get("section"),
                        "offset": m.start,
                    },
                )
            )
        return items


__all__ = ["AccessionMatcher"]
