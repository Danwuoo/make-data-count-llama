from __future__ import annotations

from typing import List

from transformers import AutoTokenizer


class TokenizerWrapper:
    """Thin wrapper around a LLaMA tokenizer for token counting."""

    def __init__(
        self, model_name: str = "hf-internal-testing/llama-tokenizer"
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens for ``text``."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> List[int]:
        """Encode ``text`` and return the token ids."""
        return self.tokenizer.encode(text, add_special_tokens=False)


__all__ = ["TokenizerWrapper"]
