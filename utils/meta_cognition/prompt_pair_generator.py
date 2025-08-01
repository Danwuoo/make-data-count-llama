from __future__ import annotations

from typing import Iterable, List, Optional

from utils.refinement.schema import CorrectionProposal

from .pair_filter import PairFilter
from .schema import ContrastivePromptPair, ErrorRecord, PromptPairItem
from .strategy_engine import PairStrategyEngine


class PromptPairGenerator:
    """High level interface to build prompt pairs from logged errors."""

    def __init__(
        self,
        strategy_engine: Optional[PairStrategyEngine] = None,
        pair_filter: Optional[PairFilter] = None,
    ) -> None:
        self.strategy_engine = strategy_engine or PairStrategyEngine()
        self.filter = pair_filter or PairFilter()

    def generate(
        self,
        errors: Iterable[ErrorRecord],
        corrections: Iterable[CorrectionProposal],
    ) -> List[PromptPairItem | ContrastivePromptPair]:
        selected = self.filter.select(errors, corrections)
        return [
            self.strategy_engine.build(err, corr)
            for err, corr in selected
        ]
