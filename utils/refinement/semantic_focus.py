import re
from collections import Counter
from typing import Iterable, List


class SemanticFocusAnalyzer:
    """Naive keyword extractor used to guide question generation."""

    def __init__(self, stop_words: Iterable[str] | None = None) -> None:
        self.stop_words = set(
            stop_words or {"the", "and", "for", "with", "that", "this", "from"}
        )

    def extract_focus_terms(self, text: str, top_k: int = 2) -> List[str]:
        """Return up to ``top_k`` keywords from ``text``."""

        tokens = re.findall(r"\w+", text.lower())
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 3]
        counts = Counter(tokens)
        return [t for t, _ in counts.most_common(top_k)]
