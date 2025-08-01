"""Split a reference block into individual citations."""

from __future__ import annotations

import re
from typing import List


class ReferenceSplitter:
    """Split references based on simple numbering patterns."""

    def split(self, text: str) -> List[str]:
        """Return a list of reference strings."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        references: List[str] = []
        current: List[str] = []
        for line in lines:
            if re.match(r"^\[?\d+\]?", line) and current:
                references.append(' '.join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            references.append(' '.join(current).strip())
        return references
