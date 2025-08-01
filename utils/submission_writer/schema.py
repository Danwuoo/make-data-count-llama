"""Schema definitions for Kaggle submission rows."""
from dataclasses import dataclass


@dataclass
class SubmissionRow:
    """Single row in the final submission file."""

    id: str
    label: str
