import uuid
from typing import Dict, List

from .semantic_focus import SemanticFocusAnalyzer
from .template_bank import QuestionTemplateBank
from .template_renderer import TemplateRenderer
from .schema import SelfQuestionItem


class SelfQuestioner:
    """Generate self-questions to trigger the refinement loop."""

    def __init__(
        self,
        template_bank: QuestionTemplateBank | None = None,
        focus_analyzer: SemanticFocusAnalyzer | None = None,
        renderer: TemplateRenderer | None = None,
    ) -> None:
        self.template_bank = template_bank or QuestionTemplateBank()
        self.focus_analyzer = focus_analyzer or SemanticFocusAnalyzer()
        self.renderer = renderer or TemplateRenderer()

    def generate(
        self, context_unit: Dict, prediction: str
    ) -> List[SelfQuestionItem]:
        """Create self-questions given a context and model prediction."""

        text = context_unit.get("text", "")
        context_id = context_unit.get("context_id", "")
        doc_id = context_unit.get("doc_id")
        section = context_unit.get("section")
        confidence = float(context_unit.get("confidence", 0.0))

        focus_terms = self.focus_analyzer.extract_focus_terms(text)
        questions: List[SelfQuestionItem] = []

        for qtype in self.template_bank.templates.keys():
            templates = self.template_bank.get_templates(qtype)
            for focus in focus_terms:
                for tmpl in templates:
                    question_text = self.renderer.render(
                        tmpl, focus=focus, prediction=prediction
                    )
                    question_id = f"qst_{uuid.uuid4().hex[:8]}"
                    questions.append(
                        SelfQuestionItem(
                            context_id=context_id,
                            question_id=question_id,
                            question_text=question_text,
                            question_type=qtype,
                            confidence_level=confidence,
                            source={
                                "doc_id": doc_id or "",
                                "section": section or "",
                                "original_prediction": prediction,
                            },
                        )
                    )
        return questions
