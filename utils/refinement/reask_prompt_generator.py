"""Utilities to build re-ask prompts combining context and self-question."""

from __future__ import annotations

from jinja2 import Template


_DEFAULT_TEMPLATE = (
    "{{context}}\n\nQuestion: {{question}}\nAnswer:"
)


class ReAskPromptGenerator:
    """Compose prompts for the second-pass reasoning step."""

    def __init__(self, template: str | None = None) -> None:
        self.template = template or _DEFAULT_TEMPLATE

    def build(self, context: str, question: str) -> str:
        """Render a prompt joining the original context and follow-up question."""

        tmpl = Template(self.template)
        return tmpl.render(context=context, question=question)
