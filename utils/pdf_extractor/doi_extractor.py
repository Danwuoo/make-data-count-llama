"""Find and normalise DOI strings within text."""

from __future__ import annotations

import re
from typing import Optional


class DOIExtractor:
    """Locate a DOI using a simple regular expression."""

    DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)

    def extract(self, text: str) -> Optional[str]:
        """Return the first DOI found in *text*, if any."""
        match = self.DOI_PATTERN.search(text)
        return match.group(0) if match else None
