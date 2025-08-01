from __future__ import annotations

"""Schemas and helpers for prompt perturbation reports."""

from dataclasses import dataclass, asdict
from typing import Iterable, List
import json


@dataclass
class VariantOutput:
    """Model output for a single perturbed prompt."""

    prompt_variant: str
    label: str
    confidence: float


@dataclass
class PerturbationReport:
    """Aggregated report summarizing perturbation test results."""

    context_id: str
    original_label: str
    perturbation_variants: int
    match_count: int
    invariance_score: float
    avg_confidence_drop: float
    variant_outputs: List[VariantOutput]
    is_consistent: bool


class ReportFormatter:
    """Utility to serialize :class:`PerturbationReport` instances."""

    def to_json(self, report: PerturbationReport) -> str:
        """Serialize ``report`` to a JSON string."""
        return json.dumps(asdict(report), ensure_ascii=False)

    def to_jsonl(self, reports: Iterable[PerturbationReport]) -> str:
        """Serialize an iterable of reports into JSON Lines format."""
        return "\n".join(self.to_json(r) for r in reports)
