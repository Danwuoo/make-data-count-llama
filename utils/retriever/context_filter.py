"""Metadata based filtering utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List


class ContextFilter:
    """Filter context metadata according to field equality."""

    @staticmethod
    def match(metadata: dict, filters: Dict[str, str]) -> bool:
        return all(metadata.get(k) == v for k, v in filters.items())

    def apply(
        self, ids: Iterable[int], metadata: List[dict], filters: Dict[str, str]
    ) -> List[int]:
        return [
            i for i in ids if self.match(metadata[i], filters)
        ]
