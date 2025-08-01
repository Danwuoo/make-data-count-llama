"""L6: Retrieval utilities for memory stores."""

from .retriever.context_retriever import ContextRetriever

__all__ = ["ContextRetriever", "retrieve"]


def retrieve(query: str) -> list[str]:
    """Placeholder retriever returning an empty list.

    The :class:`ContextRetriever` class in ``utils.retriever`` provides the
    full semantic retrieval implementation.
    """

    return []
