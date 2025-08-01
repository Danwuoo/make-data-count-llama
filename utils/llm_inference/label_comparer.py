from __future__ import annotations

"""Compare labels across perturbation runs."""

from dataclasses import dataclass
from typing import List

from .llama3_inference import LLMResult


@dataclass
class CompareResult:
    """Outcome of label comparison."""

    match_count: int
    mismatched_indices: List[int]


class LabelComparer:
    """Checks whether variant outputs align with the original label."""

    def compare(
        self,
        original_label: str,
        results: List[LLMResult],
    ) -> CompareResult:
        match_count = 0
        mismatches: List[int] = []
        for idx, res in enumerate(results):
            if res.predicted_label == original_label:
                match_count += 1
            else:
                mismatches.append(idx)
        return CompareResult(
            match_count=match_count,
            mismatched_indices=mismatches,
        )
