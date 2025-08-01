"""Enumeration of decoding strategies for :class:`LLMOutputDecoder`."""
from __future__ import annotations

from enum import Enum


class DecodingStrategy(str, Enum):
    """Supported decoding strategies for interpreting model output."""

    DIRECT_LABEL = "direct_label"
    TEXT2LABEL = "text2label"
    LOGIT_MAPPED = "logit-mapped"
