from typing import Dict, List


class QuestionTemplateBank:
    """Repository for self-question templates grouped by type."""

    def __init__(self) -> None:
        self.templates: Dict[str, List[str]] = {
            "classification_challenge": [
                (
                    "Is the reference to ${focus} truly a ${prediction} "
                    "citation, or does it play another role?"
                ),
            ],
            "semantic_contrast": [
                (
                    "How would the interpretation change if ${focus} were "
                    "considered primary evidence?"
                ),
            ],
            "detail_check": [
                "What specific data about ${focus} is being referenced here?",
            ],
            "self_reflection": [
                "Can you justify labeling ${focus} as ${prediction}?",
            ],
        }

    def get_templates(self, question_type: str) -> List[str]:
        """Return templates for a given question type."""

        return self.templates.get(question_type, [])
