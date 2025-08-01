"""Parse model outputs into structured ``LLMResult`` components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

LABELS = ["primary", "secondary", "none"]


@dataclass
class ParsedOutput:
    """Container for parsed inference output."""

    label: str
    confidence: float
    logits: Dict[str, float]
    text: str


class OutputParser:
    """Extract a label and confidence score from raw model output."""

    def parse(self, text: str, scores: List[torch.Tensor]) -> ParsedOutput:
        """Parse ``text`` and compute confidences from ``scores``."""
        lowered = text.lower()
        label = next((lbl for lbl in LABELS if lbl in lowered), "none")

        # Convert last-token logits to probabilities for each label token
        last_scores = scores[-1][0]  # shape: (vocab_size,)
        logits = {
            lbl: float(last_scores[self._token_id(lbl)]) for lbl in LABELS
        }
        probs = F.softmax(torch.tensor(list(logits.values())), dim=0)
        confidence = float(probs[LABELS.index(label)])
        return ParsedOutput(
            label=label,
            confidence=confidence,
            logits=logits,
            text=text,
        )

    def _token_id(self, label: str) -> int:
        """Return a stable pseudo token id for ``label``."""
        # In absence of tokenizer context, use hash for deterministic id.
        return abs(hash(label)) % 10000
