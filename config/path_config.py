import os
from pathlib import Path

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    Path("/kaggle/input/make-data-count-finding-data-reuse")
    if IS_KAGGLE
    else REPO_ROOT / "data"
)
TRAIN_DOCS_DIR = DATA_ROOT / "train"
TEST_DOCS_DIR = DATA_ROOT / "test"
TRAIN_LABELS_PATH = DATA_ROOT / "train_labels.csv"
SAMPLE_SUBMISSION_PATH = DATA_ROOT / "sample_submission.csv"

WORKING_OUTPUT_DIR = (
    Path("/kaggle/working/output") if IS_KAGGLE else REPO_ROOT / "output"
)


def _find_model_mount_dir() -> Path:
    """Locate a mounted model directory on Kaggle or fall back locally."""
    if IS_KAGGLE:
        kaggle_input = Path("/kaggle/input")
        for path in kaggle_input.iterdir():
            if not path.is_dir():
                continue
            markers = ["tokenizer_config.json", "tokenizer.model", "config.json"]
            if any((path / m).exists() for m in markers):
                return path
        return kaggle_input
    return REPO_ROOT / "models"


MODEL_MOUNT_DIR = _find_model_mount_dir()

ERRORS_DIR = WORKING_OUTPUT_DIR / "errors"
PREDICTIONS_DIR = WORKING_OUTPUT_DIR / "predictions"
LORA_ADAPTERS_DIR = WORKING_OUTPUT_DIR / "lora_adapters"
CORRECTIONS_LOG_PATH = PREDICTIONS_DIR / "corrections.jsonl"

__all__ = [
    "IS_KAGGLE",
    "DATA_ROOT",
    "TRAIN_DOCS_DIR",
    "TEST_DOCS_DIR",
    "TRAIN_LABELS_PATH",
    "SAMPLE_SUBMISSION_PATH",
    "WORKING_OUTPUT_DIR",
    "MODEL_MOUNT_DIR",
    "ERRORS_DIR",
    "PREDICTIONS_DIR",
    "LORA_ADAPTERS_DIR",
    "CORRECTIONS_LOG_PATH",
]
