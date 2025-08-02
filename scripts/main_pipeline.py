"""CLI entry point for running the full MDC prediction pipeline.

This script loads pre-built :class:`~utils.context_builder.schema.ContextUnit`
records and performs classification using one of the available model
backends. Low-confidence predictions can optionally be refined via a
self-questioning loop. Final predictions are written to JSONL files with
timestamps and converted into the competition submission format.

The interface is intentionally lightweight so it can be executed inside a
Kaggle notebook via ``!python scripts/main_pipeline.py`` without requiring
any multiprocessing or heavy external dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from config.path_config import MODEL_MOUNT_DIR
from utils.context_builder.schema import ContextUnit
from utils.llm_inference import LLMResult
from utils.llm_inference.inference_engine import get_model
from utils.output_writer import generate_submission
from utils.refinement import RefinementEngine


# ---------------------------------------------------------------------------
# Configuration

CONFIDENCE_THRESHOLD = 0.7


def _resolve_model_path(model_name: str) -> Path:
    """Return the filesystem path for ``model_name``.

    The repository ships with a LLaMA‑3 checkpoint under ``models/``. For other
    models we fall back to ``MODEL_MOUNT_DIR / model_name`` so users can mount
    additional weights as needed.
    """

    mapping = {
        "llama3": MODEL_MOUNT_DIR / "llama-3-8b-instruct",
        "mixtral": MODEL_MOUNT_DIR / "mixtral-8x7b-instruct",
        "qwen": MODEL_MOUNT_DIR / "qwen-7b-instruct",
        "gemma": MODEL_MOUNT_DIR / "gemma-7b-it",
        "deepseek": MODEL_MOUNT_DIR / "deepseek-7b-base",
    }
    return Path(mapping.get(model_name.lower(), MODEL_MOUNT_DIR / model_name))


# ---------------------------------------------------------------------------
# Data loading utilities

def _load_jsonl(path: Path) -> List[ContextUnit]:
    with path.open("r", encoding="utf-8") as fh:
        return [ContextUnit.model_validate_json(line) for line in fh if line.strip()]


def _load_parquet(path: Path) -> List[ContextUnit]:
    import pandas as pd  # imported lazily to keep dependencies minimal

    df = pd.read_parquet(path)
    return [ContextUnit(**row) for row in df.to_dict(orient="records")]


def load_contexts(input_path: str) -> List[ContextUnit]:
    """Load context units from ``input_path`` (JSONL or Parquet)."""

    path = Path(input_path)
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return _load_parquet(path)
    raise ValueError("Unsupported input format: expected JSONL or Parquet")


# ---------------------------------------------------------------------------
# Core pipeline

def run_pipeline(
    contexts: Iterable[ContextUnit],
    *,
    model_name: str,
    enable_reask: bool,
    output_csv: Path,
    save_errors: bool,
) -> None:
    """Run inference, optional refinement and submission generation."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    predictions_dir = Path("data/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / f"predictions_{timestamp}.jsonl"
    corrections_path = predictions_dir / f"corrections_{timestamp}.jsonl"

    errors_path: Path | None = None
    if save_errors:
        errors_dir = Path("data/errors")
        errors_dir.mkdir(parents=True, exist_ok=True)
        errors_path = errors_dir / f"errors_{timestamp}.jsonl"

    model_path = _resolve_model_path(model_name)
    model = get_model(model_name, model_path=model_path)
    refinement = RefinementEngine() if enable_reask else None

    predictions: List[dict] = []
    corrections: List[dict] = []
    errors: List[dict] = []

    for ctx in contexts:
        result: LLMResult = model.predict(context_id=ctx.context_id, context=ctx.text)
        pred = {
            "context_id": result.context_id,
            "final_label": result.predicted_label,
            "confidence": result.confidence,
            "raw_output": result.raw_output,
            "used_strategy": result.meta.get("used_strategy", ""),
            "label_source": result.meta.get("label_source", ""),
            "logits": result.logits,
        }

        low_conf = result.confidence < CONFIDENCE_THRESHOLD

        if enable_reask and low_conf and refinement is not None:
            proposals = refinement.run(ctx.model_dump(), result)
            for proposal in proposals:
                corrections.append(asdict(proposal))
                if proposal.accepted:
                    pred["final_label"] = proposal.corrected_label
                    pred["confidence"] = proposal.corrected_confidence
                    break

        if save_errors and low_conf:
            errors.append(
                {
                    "context_id": ctx.context_id,
                    "predicted_label": result.predicted_label,
                    "confidence": result.confidence,
                }
            )

        predictions.append(pred)

    with predictions_path.open("w", encoding="utf-8") as fh:
        for p in predictions:
            fh.write(json.dumps(p) + "\n")

    if corrections:
        with corrections_path.open("w", encoding="utf-8") as fh:
            for c in corrections:
                fh.write(json.dumps(c) + "\n")

    if errors_path and errors:
        with errors_path.open("w", encoding="utf-8") as fh:
            for e in errors:
                fh.write(json.dumps(e) + "\n")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    generate_submission(predictions_path, output_csv)
    logging.info("Submission written to %s", output_csv)


# ---------------------------------------------------------------------------
# CLI entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MDC prediction pipeline")
    parser.add_argument(
        "--input",
        default="data/context/context.jsonl",
        help="Path to context units file (JSONL or Parquet)",
    )
    parser.add_argument("--model", default="llama3", help="Model backend name")
    parser.add_argument(
        "--reask",
        action="store_true",
        help="Enable self-questioning refinement for low-confidence samples",
    )
    parser.add_argument(
        "--output",
        default="data/submission/submission.csv",
        help="Path to write the final submission CSV",
    )
    parser.add_argument(
        "--save-errors",
        action="store_true",
        help="Save low-confidence samples to data/errors/",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    contexts = load_contexts(args.input)
    logging.info("Loaded %d context units", len(contexts))

    run_pipeline(
        contexts,
        model_name=args.model,
        enable_reask=args.reask,
        output_csv=Path(args.output),
        save_errors=args.save_errors,
    )


if __name__ == "__main__":
    main()

