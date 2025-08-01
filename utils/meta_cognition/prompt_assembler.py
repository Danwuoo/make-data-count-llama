from __future__ import annotations


class PromptAssembler:
    """Utility to compose prompt text from components."""

    def assemble_input(
        self,
        context: str,
        question: str | None = None,
        answer: str | None = None,
    ) -> str:
        parts: list[str] = []
        if question and answer:
            parts.append(f"Q: {question}\nA: {answer}")
        if context:
            parts.append(context)
        return "\n\n".join(parts).strip()
