from __future__ import annotations

"""Execute inference across multiple prompt variants."""

from typing import List

from .llama3_inference import LLaMA3Inference, LLMResult
from .output_decoder import DecodingStrategy


class MultiPromptRunner:
    """Runs a batch of prompts through :class:`LLaMA3Inference`."""

    def __init__(self, inference: LLaMA3Inference) -> None:
        self.inference = inference

    def run(
        self,
        context_id: str,
        prompt_variants: List[str],
    ) -> List[LLMResult]:
        """Run inference for each prompt variant and collect results."""
        results: List[LLMResult] = []
        for idx, prompt in enumerate(prompt_variants):
            tokens = self.inference.tokenizer.encode(prompt)
            generated = self.inference.engine.generate(
                tokens, max_new_tokens=32, temperature=0.0
            )
            text = self.inference.tokenizer.decode(generated["tokens"])
            prediction = self.inference.decoder.decode(
                context_id=f"{context_id}_v{idx}",
                text=text,
                scores=generated["scores"],
                strategy=DecodingStrategy.TEXT2LABEL,
            )
            results.append(
                LLMResult(
                    context_id=context_id,
                    predicted_label=prediction.final_label,
                    confidence=prediction.confidence,
                    raw_output=prediction.raw_output,
                    prompt=prompt,
                    logits=prediction.logits,
                    meta={
                        "label_source": prediction.label_source,
                        "used_strategy": prediction.used_strategy,
                    },
                )
            )
        return results
