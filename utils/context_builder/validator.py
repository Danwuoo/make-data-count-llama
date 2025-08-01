from __future__ import annotations

from typing import Iterable, List

from .schema import ContextUnit


def validate_contexts(
    contexts: Iterable[ContextUnit], max_tokens: int
) -> List[ContextUnit]:
    """Filter invalid context units."""
    valid: List[ContextUnit] = []
    for ctx in contexts:
        if not ctx.text.strip():
            continue
        if ctx.token_count > max_tokens:
            continue
        valid.append(ctx)
    return valid


__all__ = ["validate_contexts"]
