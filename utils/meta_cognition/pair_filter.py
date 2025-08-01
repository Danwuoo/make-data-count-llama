from __future__ import annotations

from typing import Iterable, List, Tuple

from utils.refinement.schema import CorrectionProposal

from .schema import ErrorRecord


class PairFilter:
    """Select valid error-correction pairs for prompt generation."""

    def __init__(
        self,
        min_confidence_delta: float = 0.0,
        require_label_flip: bool = True,
    ) -> None:
        self.min_confidence_delta = min_confidence_delta
        self.require_label_flip = require_label_flip

    def select(
        self,
        errors: Iterable[ErrorRecord],
        corrections: Iterable[CorrectionProposal],
    ) -> List[Tuple[ErrorRecord, CorrectionProposal]]:
        corr_map = {c.context_id: c for c in corrections if c.accepted}
        selected: List[Tuple[ErrorRecord, CorrectionProposal]] = []
        for err in errors:
            corr = corr_map.get(err.context_id)
            if corr is None:
                continue
            if (
                self.require_label_flip
                and corr.corrected_label == err.predicted_label
            ):
                continue
            if corr.confidence_delta < self.min_confidence_delta:
                continue
            selected.append((err, corr))
        return selected
