from utils.retriever.knn_ranker import KNNRanker
from utils.retriever.penalty_rule import ContextPenaltyRule
from utils.retriever.schema import RetrievalResultItem


def test_knn_ranker_basic_ranking():
    items = [
        RetrievalResultItem(query_text="q", matched_context="c1", similarity_score=0.8),
        RetrievalResultItem(query_text="q", matched_context="c2", similarity_score=0.9),
    ]
    ranker = KNNRanker()
    ranked = ranker.rank(items)
    assert [r.context for r in ranked] == ["c2", "c1"]
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2


def test_knn_ranker_with_penalty():
    items = [
        RetrievalResultItem(
            query_text="q", matched_context="c1", similarity_score=0.9, section="abstract"
        ),
        RetrievalResultItem(
            query_text="q", matched_context="c2", similarity_score=0.8, section="methods"
        ),
    ]
    penalty_rule = ContextPenaltyRule(penalties={"section": {"methods": -0.2}})
    ranker = KNNRanker(penalty_rule=penalty_rule)
    ranked = ranker.rank(items)
    assert ranked[0].context == "c1"
    assert ranked[1].context == "c2"
