import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from utils.context_builder import build_context  # noqa: E402
from utils.parsed_doc import ParsedDoc  # noqa: E402


def test_build_context_basic():
    doc = ParsedDoc(
        doc_id="DOC1",
        source_type="pdf",
        title="Sample Title",
        abstract="This is the abstract. It has two sentences.",
        body="Paragraph one. Still first paragraph.\nParagraph two is here.",
    )
    contexts = build_context(doc, max_tokens=20, stride=5)
    assert contexts, "No contexts returned"
    assert contexts[0].source.section == "intro"
    assert contexts[0].source.source_type == "title+abstract"
    for ctx in contexts:
        assert ctx.doc_id == "DOC1"
    for ctx in contexts[1:]:
        assert ctx.token_count <= 20
        assert ctx.source.section in {"title", "abstract", "body"}
