"""L5: Meta-training loop for continual learning.

This module wires together the components that make up the
meta-cognition layer.  It provides a small ``run_meta_loop`` helper that
logs error samples, turns them into prompt pairs and (optionally) performs
LoRA fine-tuning on those pairs.

The function is intentionally lightweight – it does not aim to be a full
pipeline manager but rather a reference implementation that can be
expanded upon.  When a ``model`` and ``tokenizer`` are not supplied the
fine‑tuning step is skipped which makes the function convenient to use in
tests or smaller workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from utils.meta_cognition import (
    ErrorLogger,
    ErrorRecord,
    LoRAFineTuner,
    PromptPairGenerator,
)
from utils.meta_cognition.error_storage import ErrorStorageManager
from utils.refinement.schema import CorrectionProposal
from config.path_config import ERRORS_DIR, LORA_ADAPTERS_DIR


def run_meta_loop(
    errors: Iterable[ErrorRecord],
    corrections: Iterable[CorrectionProposal],
    model=None,
    tokenizer=None,
    *,
    error_dir: str | Path = ERRORS_DIR,
    adapter_dir: str | Path = LORA_ADAPTERS_DIR,
    **tuner_kwargs,
) -> Tuple[list, dict]:
    """Execute the meta-learning cycle.

    Parameters
    ----------
    errors, corrections:
        Iterables of :class:`ErrorRecord` and
        :class:`~utils.refinement.schema.CorrectionProposal` respectively.
    model, tokenizer:
        When provided the function will fine‑tune ``model`` using the
        generated prompt pairs.  If either is ``None`` the fine‑tuning step
        is skipped.
    error_dir, adapter_dir:
        Directories used for error persistence and saving LoRA adapters.
    tuner_kwargs:
        Additional keyword arguments forwarded to
        :meth:`LoRAFineTuner.train`.

    Returns
    -------
    Tuple[List[PromptPairItem | ContrastivePromptPair], Dict[str, float]]
        A tuple containing the generated prompt pairs and evaluation
        metrics (empty when fine‑tuning is skipped).
    """

    logger = ErrorLogger(storage=ErrorStorageManager(error_dir))
    for record in errors:
        logger.log(record)

    generator = PromptPairGenerator()
    pairs = generator.generate(errors, corrections)

    metrics = {}
    if model is not None and tokenizer is not None and LoRAFineTuner is not None:
        from utils.meta_cognition.lora_model_manager import LoRAModelManager

        manager = LoRAModelManager(Path(adapter_dir))
        tuner = LoRAFineTuner(model, tokenizer, manager)
        trainer = tuner.train(pairs, **tuner_kwargs)
        metrics = tuner.evaluate(trainer, pairs)

    return pairs, metrics

