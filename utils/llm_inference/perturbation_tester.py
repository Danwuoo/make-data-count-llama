from __future__ import annotations

"""Main entry point for prompt perturbation robustness testing."""

from .llama3_inference import LLaMA3Inference
from .perturbation_generator import PerturbationGenerator, GeneratorConfig
from .multiprompt_runner import MultiPromptRunner
from .label_comparer import LabelComparer
from .score_computer import ScoreComputer
from .report_schema import (
    PerturbationReport,
    VariantOutput,
)


class PromptPerturbationTester:
    """Run invariance checks by perturbing prompts and evaluating outputs."""

    def __init__(
        self,
        inference: LLaMA3Inference,
        num_variants: int = 5,
        min_invariance_score: float = 0.9,
        confidence_drop_threshold: float = 0.2,
    ) -> None:
        self.generator = PerturbationGenerator(
            GeneratorConfig(num_variants=num_variants)
        )
        self.runner = MultiPromptRunner(inference)
        self.comparer = LabelComparer()
        self.scorer = ScoreComputer()
        self.min_invariance_score = min_invariance_score
        self.confidence_drop_threshold = confidence_drop_threshold

    def test(
        self,
        context_id: str,
        prompt: str,
        original_label: str,
        original_confidence: float,
    ) -> PerturbationReport:
        """Execute the perturbation test and return a structured report."""
        variants = self.generator.generate(prompt)
        results = self.runner.run(context_id, variants)
        compare = self.comparer.compare(original_label, results)
        score = self.scorer.compute(
            original_confidence, original_label, results
        )
        variant_outputs = [
            VariantOutput(
                prompt_variant=variants[i],
                label=r.predicted_label,
                confidence=r.confidence,
            )
            for i, r in enumerate(results)
        ]
        is_consistent = (
            score.invariance_score >= self.min_invariance_score
            and score.avg_confidence_drop <= self.confidence_drop_threshold
        )
        return PerturbationReport(
            context_id=context_id,
            original_label=original_label,
            perturbation_variants=len(variants),
            match_count=compare.match_count,
            invariance_score=score.invariance_score,
            avg_confidence_drop=score.avg_confidence_drop,
            variant_outputs=variant_outputs,
            is_consistent=is_consistent,
        )
