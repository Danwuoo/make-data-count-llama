"""L4: Self-refinement loop utilities."""

from .self_questioner import SelfQuestioner


def refine_prediction(prediction: str) -> str:
    """Placeholder refinement that echoes the prediction."""
    return prediction


__all__ = ["SelfQuestioner", "refine_prediction"]
