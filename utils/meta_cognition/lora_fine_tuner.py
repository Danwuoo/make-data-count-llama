from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Type

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .data_preprocessor import DataPreprocessor
from .eval_monitor import EvalMonitor
from .lora_model_manager import LoRAModelManager
from .schema import PromptPairItem
from .trainer_builder import TrainerBuilder


@dataclass
class LoRAFineTuner:
    """High level interface to perform LoRA fine-tuning on prompt pairs."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model_manager: LoRAModelManager
    trainer_builder_cls: Type[TrainerBuilder] = TrainerBuilder
    data_preprocessor_cls: Type[DataPreprocessor] = DataPreprocessor
    eval_monitor: EvalMonitor = EvalMonitor()

    def train(self, pairs: Iterable[PromptPairItem], **kwargs):
        preprocessor = self.data_preprocessor_cls(self.tokenizer)
        dataset = preprocessor.to_dataset(pairs)
        builder = self.trainer_builder_cls(self.model, self.tokenizer)
        trainer = builder.build(train_dataset=dataset, **kwargs)
        trainer.train()
        version = kwargs.get("version", "latest")
        self.model_manager.save(trainer.model, version)
        return trainer

    def evaluate(self, trainer, pairs: Iterable[PromptPairItem]):
        preprocessor = self.data_preprocessor_cls(self.tokenizer)
        dataset = preprocessor.to_dataset(pairs)
        return self.eval_monitor.evaluate(trainer, dataset)
