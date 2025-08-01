"""L4: Self-refinement loop utilities."""

from .self_questioner import SelfQuestioner
from .self_corrector import SelfCorrector


def refine_prediction(prediction: str) -> str:
    """Placeholder refinement that echoes the prediction."""
    return prediction


__all__ = ["SelfQuestioner", "SelfCorrector", "refine_prediction"]
