from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .schema import PromptPairItem


@dataclass
class DataPreprocessor:
    """Convert :class:`PromptPairItem` into tokenized datasets."""

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 2048

    def to_dataset(self, pairs: Iterable[PromptPairItem]) -> Dataset:
        """Build a :class:`datasets.Dataset` for language model tuning."""

        def gen():
            for item in pairs:
                prompt = item.instruction
                if item.input:
                    prompt += "\n" + item.input
                yield {"prompt": prompt, "response": item.output}

        dataset = Dataset.from_generator(gen)

        def tokenize(batch):
            sources = self.tokenizer(
                batch["prompt"], truncation=True, max_length=self.max_length
            )
            targets = self.tokenizer(
                batch["response"], truncation=True, max_length=self.max_length
            )
            return {
                "input_ids": sources["input_ids"],
                "labels": targets["input_ids"],
            }

        return dataset.map(
            tokenize, batched=True, remove_columns=["prompt", "response"]
        )
