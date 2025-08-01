"""Decode raw inference output into structured predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from .confidence_scorer import ConfidenceScorer
from .decoding_strategy import DecodingStrategy
from .label_extractor import LabelExtractor
from .logit_decoder import LogitDecoder
from .validator import LabelValidator


@dataclass
class FinalPrediction:
    """Standardized prediction structure produced by the decoder."""

    context_id: str
    final_label: str
    confidence: float
    raw_output: str
    used_strategy: str
    label_source: str
    logits: Dict[str, float]


class LLMOutputDecoder:
    """High-level decoder combining text and logit cues."""

    def __init__(self, labels: Sequence[str] | None = None, min_confidence: float = 0.0) -> None:
        self.labels = list(labels or ["primary", "secondary", "none"])
        self.extractor = LabelExtractor(self.labels)
        self.logit_decoder = LogitDecoder(self.labels)
        self.scorer = ConfidenceScorer()
        self.validator = LabelValidator(self.labels, min_confidence)

    def decode(
        self,
        *,
        context_id: str,
        text: str,
        scores: Sequence[Sequence[float]],
        strategy: DecodingStrategy = DecodingStrategy.TEXT2LABEL,
    ) -> FinalPrediction:
        """Decode model ``text`` and ``scores`` into a :class:`FinalPrediction`."""

        probs, logits = self.logit_decoder.decode(scores)
        logit_label = max(probs, key=probs.get)
        text_label, source = self.extractor.extract(text)

        if strategy == DecodingStrategy.LOGIT_MAPPED:
            final_label = logit_label
            label_source = "logit"
        elif strategy == DecodingStrategy.DIRECT_LABEL:
            if text_label in self.labels:
                final_label = text_label
                label_source = source
            else:
                final_label = logit_label
                label_source = "logit"
        else:  # TEXT2LABEL
            final_label = text_label
            label_source = source

        confidence = self.scorer.compute(probs, final_label)
        self.validator.validate(final_label, confidence)

        return FinalPrediction(
            context_id=context_id,
            final_label=final_label,
            confidence=confidence,
            raw_output=text,
            used_strategy=strategy.value,
            label_source=label_source,
            logits=logits,
        )
