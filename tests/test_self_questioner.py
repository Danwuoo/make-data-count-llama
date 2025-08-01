from utils.refinement import SelfQuestioner
from utils.refinement.schema import SelfQuestionItem


def test_generate_questions() -> None:
    questioner = SelfQuestioner()
    context = {
        "context_id": "ctx1",
        "text": "This sentence refers to the XYZ dataset for comparison.",
        "doc_id": "DOC1",
        "section": "results",
        "confidence": 0.52,
    }
    prediction = "secondary"

    questions = questioner.generate(context, prediction)

    assert questions, "No questions generated"
    assert isinstance(questions[0], SelfQuestionItem)
