"""Utilities for semantic context retrieval."""

try:  # pragma: no cover - optional dependencies
    from .context_retriever import ContextRetriever
    from .embedding_encoder import EmbeddingEncoder
    from .vector_indexer import VectorIndexer
    from .index_storage import IndexStorageManager
    from .context_filter import ContextFilter
except Exception:  # pragma: no cover - optional dependencies
    ContextRetriever = EmbeddingEncoder = VectorIndexer = IndexStorageManager = ContextFilter = None

from .knn_ranker import KNNRanker
from .penalty_rule import ContextPenaltyRule
from .ranked_context_builder import RankedContextBuilder
from .reranker_engine import RerankerEngine
from .score_combiner import ScoreCombiner
from .schema import RankedContext, RetrievalResultItem

__all__ = [
    "ContextRetriever",
    "EmbeddingEncoder",
    "VectorIndexer",
    "IndexStorageManager",
    "ContextFilter",
    "KNNRanker",
    "ContextPenaltyRule",
    "RankedContextBuilder",
    "RerankerEngine",
    "ScoreCombiner",
    "RetrievalResultItem",
    "RankedContext",
]
