"""Wrapper around ``AutoTokenizer`` providing common options."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from transformers import AutoTokenizer


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer behaviour."""

    model_path: str
    padding: bool = True
    truncation: bool = True
    max_length: int | None = None


class TokenizerWrapper:
    """Thin wrapper that exposes ``encode`` and ``decode`` methods."""

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    def encode(self, text: str) -> Dict[str, Any]:
        """Tokenize ``text`` according to the configuration."""
        return self.tokenizer.batch_encode_plus(
            [text],
            padding=self.config.padding,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    def decode(self, token_ids) -> str:
        """Decode token ids back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
