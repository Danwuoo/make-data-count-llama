"""Utilities for constructing prompts for LLaMA 3 inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PromptTemplate:
    """Container for a prompt template."""

    name: str
    template: str

    def render(self, context: str) -> str:
        """Merge ``context`` into the template."""
        return self.template.format(context=context)


class PromptGenerator:
    """Generate prompts for different inference strategies."""

    _TEMPLATES: Dict[str, PromptTemplate] = {
        "zero-shot": PromptTemplate(
            name="zero-shot",
            template=(
                "You are a citation classifier. "
                "Classify the following text as primary, secondary, or none.\n"
                "Text: {context}\nLabel:"
            ),
        ),
        "few-shot": PromptTemplate(
            name="few-shot",
            template=(
                "You are a citation classifier.\n"
                "Example: Text: 'Data were collected from surveys.' "
                "Label: primary\n"
                "Example: Text: 'We refer to CDC statistics.' "
                "Label: secondary\n"
                "Now classify the following text.\n"
                "Text: {context}\nLabel:"
            ),
        ),
        "cot-style": PromptTemplate(
            name="cot-style",
            template=(
                "Classify the citation as primary, secondary, or none. "
                "Think step by step before giving the final answer.\n"
                "Text: {context}\nReasoning:"
            ),
        ),
    }

    def generate(self, context: str, strategy: str = "zero-shot") -> str:
        """Return a prompt for ``context`` using ``strategy``."""
        template = self._TEMPLATES.get(strategy)
        if not template:
            raise ValueError(f"Unknown strategy: {strategy}")
        return template.render(context)
