"""Rank and filter retrieved contexts based on multiple scoring strategies."""

from __future__ import annotations

from typing import List, Sequence

from .penalty_rule import ContextPenaltyRule
from .ranked_context_builder import RankedContextBuilder
from .reranker_engine import RerankerEngine
from .score_combiner import ScoreCombiner
from .schema import RankedContext, RetrievalResultItem


class KNNRanker:
    """Re-rank contexts returned by :class:`ContextRetriever`.

    The ranker supports combining similarity scores with optional reranker
    scores and metadata-based penalties.
    """

    def __init__(
        self,
        *,
        combiner: ScoreCombiner | None = None,
        reranker: RerankerEngine | None = None,
        penalty_rule: ContextPenaltyRule | None = None,
        top_k: int | None = None,
    ) -> None:
        self.combiner = combiner or ScoreCombiner()
        self.reranker = reranker
        self.penalty_rule = penalty_rule or ContextPenaltyRule()
        self.builder = RankedContextBuilder()
        self.top_k = top_k

    def rank(self, retrieval_results: Sequence[RetrievalResultItem]) -> List[RankedContext]:
        """Return contexts ordered by their final ranking score."""

        scored: List[tuple[float, RetrievalResultItem, float | None, float]] = []
        for item in retrieval_results:
            rerank_score = (
                self.reranker.score(item.query_text, item.matched_context)
                if self.reranker
                else None
            )
            base_score = self.combiner.combine(
                {
                    "similarity": item.similarity_score,
                    "rerank": rerank_score or 0.0,
                }
            )
            penalty = self.penalty_rule.apply(item.dict())
            final_score = base_score + penalty
            scored.append((final_score, item, rerank_score, penalty))

        scored.sort(key=lambda x: x[0], reverse=True)
        limit = self.top_k or len(scored)
        ranked_results: List[RankedContext] = []
        for rank_idx, (score, item, rerank_score, penalty) in enumerate(
            scored[:limit], start=1
        ):
            metadata = {
                "source": "retrieval" + (" + rerank" if rerank_score is not None else ""),
                "penalty": penalty,
                "weights": self.combiner.weights,
            }
            ranked_results.append(
                self.builder.build(
                    query_text=item.query_text,
                    context=item.matched_context,
                    rank=rank_idx,
                    similarity_score=item.similarity_score,
                    final_score=score,
                    reranker_score=rerank_score,
                    doc_id=item.doc_id,
                    context_id=item.context_id,
                    section=item.section,
                    metadata=metadata,
                )
            )
        return ranked_results
