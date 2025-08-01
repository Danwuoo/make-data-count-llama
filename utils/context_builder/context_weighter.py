from __future__ import annotations


def positional_weight(start_idx: int, total_sentences: int) -> float:
    """Simple position-based weighting.

    Earlier sentences receive higher scores.
    """
    if total_sentences == 0:
        return 0.0
    return 1.0 - (start_idx / total_sentences)


def compute_importance(
    start_idx: int, total_sentences: int, section: str
) -> float:
    """Combine positional weight with section heuristics."""
    weight = positional_weight(start_idx, total_sentences)
    if section == "title":
        weight += 0.5
    elif section == "abstract":
        weight += 0.25
    return min(weight, 1.0)


__all__ = ["compute_importance"]
