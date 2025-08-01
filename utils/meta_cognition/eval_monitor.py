from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from transformers import Trainer


@dataclass
class EvalMonitor:
    """Simple evaluation helper for fine-tuned models."""

    def accuracy(self, preds: Iterable[str], labels: Iterable[str]) -> float:
        pairs = list(zip(preds, labels))
        if not pairs:
            return 0.0
        correct = sum(p == l for p, l in pairs)
        return correct / len(pairs)

    def evaluate(self, trainer: Trainer, dataset) -> dict:
        """Run evaluation using the provided trainer and dataset."""
        metrics = trainer.evaluate(dataset)
        predictions = trainer.predict(dataset)
        decoded_preds = trainer.tokenizer.batch_decode(
            predictions.predictions, skip_special_tokens=True
        )
        decoded_labels = trainer.tokenizer.batch_decode(
            dataset["labels"], skip_special_tokens=True
        )
        metrics["accuracy"] = self.accuracy(decoded_preds, decoded_labels)
        return metrics
