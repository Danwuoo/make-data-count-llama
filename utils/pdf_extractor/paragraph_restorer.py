"""Clean and reconstruct paragraphs from raw PDF text."""

from __future__ import annotations

from typing import List


class ParagraphRestorer:
    """Remove spurious line breaks and rebuild paragraphs."""

    def restore(self, text: str) -> List[str]:
        """Return a list of paragraphs from raw *text*."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        paragraphs: List[str] = []
        buffer: List[str] = []
        for line in lines:
            buffer.append(line)
            if line.endswith(('.', '!', '?')):
                paragraphs.append(' '.join(buffer))
                buffer = []
        if buffer:
            paragraphs.append(' '.join(buffer))
        return paragraphs
