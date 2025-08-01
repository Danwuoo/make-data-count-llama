from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from peft import LoraConfig, get_peft_model
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainerBuilder:
    """Build a Trainer configured for LoRA fine-tuning."""

    base_model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    peft_config: Optional[LoraConfig] = None

    def build(self, train_dataset, eval_dataset=None, **kwargs) -> Trainer:
        config = self.peft_config or LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(self.base_model, config)
        args = kwargs.get("args") or TrainingArguments(
            output_dir="lora_outputs",
            per_device_train_batch_size=1,
            num_train_epochs=1,
            learning_rate=5e-5,
            logging_steps=10,
        )
        return Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
