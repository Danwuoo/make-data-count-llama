from dataclasses import dataclass
from typing import Any, Dict, List
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from utils.classifier import LLMClassifier  # noqa: E402


@dataclass
class DummyResult:
    context_id: str
    predicted_label: str
    confidence: float
    raw_output: str
    prompt: str
    logits: Dict[str, float]
    meta: Dict[str, Any]


class DummyInference:
    def infer(
        self, context_id: str, context: str, strategy: str = "zero-shot"
    ) -> DummyResult:
        return DummyResult(
            context_id=context_id,
            predicted_label="primary",
            confidence=0.9,
            raw_output="This is a primary citation.",
            prompt="prompt",
            logits={"primary": 1.0, "secondary": -1.0, "none": -2.0},
            meta={
                "used_strategy": "text2label",
                "label_source": "direct_label",
            },
        )


@dataclass
class DummyReport:
    context_id: str
    original_label: str
    perturbation_variants: int
    match_count: int
    invariance_score: float
    avg_confidence_drop: float
    variant_outputs: List[Any]
    is_consistent: bool


class DummyTester:
    def test(
        self,
        context_id: str,
        prompt: str,
        original_label: str,
        original_confidence: float,
    ) -> DummyReport:
        return DummyReport(
            context_id=context_id,
            original_label=original_label,
            perturbation_variants=1,
            match_count=1,
            invariance_score=0.96,
            avg_confidence_drop=0.0,
            variant_outputs=[],
            is_consistent=True,
        )


def test_classifier_with_perturbation():
    classifier = LLMClassifier(
        inference=DummyInference(), tester=DummyTester()
    )
    prediction = classifier.classify(
        "some text", context_id="ctx1"
    )
    assert prediction.final_label == "primary"
    assert prediction.is_consistent is True
    assert prediction.perturbation_score == 0.96
