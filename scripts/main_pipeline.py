from dataclasses import asdict
import argparse
import json

from utils.parser import parse_document
from utils.context_builder import build_context
from utils.classifier import LLMClassifier
from utils.llm_inference import LLMResult, FinalPrediction
from utils.refinement import RefinementEngine
from utils.meta_cognition import ErrorRecord, ErrorType
from utils.meta_loop import run_meta_loop
from utils.retriever import MemoryBuilder, ContextRetriever
from utils.output_writer import generate_submission


def run_pipeline(input_path: str, predictions_path: str, submission_path: str) -> None:
    """Run the SEL pipeline end to end."""
    # Parse PDF/XML into ParsedDoc
    doc = parse_document(input_path)

    # Build context units
    contexts = build_context(doc)

    # Build retrieval memory
    mem_builder = MemoryBuilder()
    indexer, metadata = mem_builder.build([c.model_dump() for c in contexts])
    retriever = ContextRetriever(mem_builder.encoder, indexer, metadata)

    clf = LLMClassifier()
    refinement = RefinementEngine()
    preds, errors, corrections = [], [], []
    CONF_THRESHOLD = 0.7
    for ctx in contexts:
        examples = [
            r.matched_context
            for r in retriever.retrieve(ctx.text, top_k=3)
            if r.context_id != ctx.context_id
        ]
        few_shot = "\n\n".join(examples)
        text = f"{few_shot}\n\n{ctx.text}" if few_shot else ctx.text
        result: LLMResult = clf.inference.infer(context_id=ctx.context_id, context=text)
        pred = FinalPrediction(
            context_id=result.context_id,
            final_label=result.predicted_label,
            confidence=result.confidence,
            raw_output=result.raw_output,
            used_strategy=result.meta.get("used_strategy", ""),
            label_source=result.meta.get("label_source", ""),
            logits=result.logits,
        )
        low_conf = pred.confidence < CONF_THRESHOLD
        inconsistent = pred.is_consistent is False
        if low_conf or inconsistent:
            proposals = refinement.run(ctx.model_dump(), result)
            corrections.extend(proposals)
            err_type = ErrorType.LOW_CONFIDENCE if low_conf else ErrorType.INCONSISTENT_OUTPUT
            errors.append(
                ErrorRecord(
                    error_id=f"err_{ctx.context_id}",
                    context_id=ctx.context_id,
                    error_type=err_type,
                    source_module="LLMClassifier",
                    original_label=result.predicted_label,
                    confidence=result.confidence,
                    confidence_threshold=CONF_THRESHOLD,
                    reason="confidence below threshold" if low_conf else "prediction inconsistent",
                )
            )
            for prop in proposals:
                if prop.accepted:
                    if prop.corrected_label != result.predicted_label:
                        errors.append(
                            ErrorRecord(
                                error_id=f"err_corr_{ctx.context_id}",
                                context_id=ctx.context_id,
                                error_type=ErrorType.CLASSIFICATION_ERROR,
                                source_module="RefinementEngine",
                                original_label=result.predicted_label,
                                refined_label=prop.corrected_label,
                                confidence=prop.corrected_confidence,
                                confidence_threshold=CONF_THRESHOLD,
                                reason=prop.correction_reason,
                            )
                        )
                    pred.final_label = prop.corrected_label
                    pred.confidence = prop.corrected_confidence
                    break
        preds.append(asdict(pred))

    run_meta_loop(errors, corrections)

    # Write predictions and final submission
    with open(predictions_path, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p) + "\n")
    generate_submission(predictions_path, submission_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SEL pipeline")
    parser.add_argument("--input", required=True, help="Path to input PDF or XML file")
    parser.add_argument("--predictions", required=True, help="Path to write predictions JSONL")
    parser.add_argument("--submission", required=True, help="Path to write submission CSV")
    args = parser.parse_args()
    run_pipeline(args.input, args.predictions, args.submission)


if __name__ == "__main__":
    main()
