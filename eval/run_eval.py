#!/usr/bin/env python3
"""
Main evaluation runner for VLM model comparison.

Runs evaluation on a model (baseline or fine-tuned) and saves results.
Supports both automated metrics and LLM-as-judge evaluation.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import VLMEvaluator, MetricResult
from eval.model_wrapper import create_wrapper


def load_eval_set(path: Path) -> list[dict]:
    """Load evaluation set from JSONL file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_question_answer(record: dict) -> tuple[str, str]:
    """Extract question and reference answer from record."""
    conversations = record.get("conversations", [])
    question = ""
    answer = ""

    for turn in conversations:
        if turn.get("role") == "user":
            content = turn.get("content", "")
            # Handle both string and list content formats
            if isinstance(content, str):
                question = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        question = item.get("text", "")
        elif turn.get("role") == "assistant":
            answer = turn.get("content", "")

    return question, answer


def run_evaluation(
    model_wrapper,
    eval_set: list[dict],
    evaluator: VLMEvaluator,
    image_base_path: str,
    use_llm_judge: bool = False,
    max_samples: int = None,
    system_prompt: str = None,
) -> dict:
    """
    Run evaluation on all samples in eval_set.

    Args:
        model_wrapper: VLM model wrapper for inference.
        eval_set: List of evaluation examples.
        evaluator: VLMEvaluator instance.
        image_base_path: Base path for image files.
        use_llm_judge: Whether to run LLM-as-judge metrics.
        max_samples: Optional limit on number of samples.
        system_prompt: Optional system prompt for model.

    Returns:
        Dict containing all evaluation results.
    """
    results = {
        "model_info": model_wrapper.get_model_info(),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "image_base_path": image_base_path,
            "use_llm_judge": use_llm_judge,
            "max_samples": max_samples,
        },
        "samples": [],
        "aggregate_metrics": {},
        "by_question_type": {},
        "by_content_type": {},
    }

    samples = eval_set[:max_samples] if max_samples else eval_set

    print(f"\nRunning evaluation on {len(samples)} samples...")
    print(f"Model: {model_wrapper.get_model_info()['model_path']}")

    all_metric_results = []
    start_time = time.time()

    for i, record in enumerate(tqdm(samples, desc="Evaluating")):
        # Extract data from record
        question, reference = extract_question_answer(record)
        image_path = record.get("image", "")
        metadata = record.get("metadata", {})

        question_type = metadata.get("question_type", "unknown")
        content_type = metadata.get("content_type", "unknown")

        # Resolve image path
        if image_base_path and not os.path.isabs(image_path):
            full_image_path = os.path.join(image_base_path, image_path)
        else:
            full_image_path = image_path

        # Generate model prediction
        try:
            prediction = model_wrapper.generate(
                image_path=full_image_path,
                question=question,
                system_prompt=system_prompt,
            )
        except Exception as e:
            prediction = f"[ERROR: {str(e)}]"
            print(f"\nError on sample {i}: {e}")

        # Run metrics
        keywords = metadata.get("keywords", [])
        is_safety = question_type == "safety" or metadata.get("is_critical", False)

        metric_results = evaluator.evaluate(
            prediction=prediction,
            reference=reference,
            question=question,
            question_type=question_type,
            content_type=content_type,
            keywords=keywords,
            is_safety_critical=is_safety,
        )

        all_metric_results.append(metric_results)

        # Store sample result
        sample_result = {
            "index": i,
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "question_type": question_type,
            "content_type": content_type,
            "image": image_path,
            "metadata": metadata,
            "metrics": {
                name: {
                    "score": r.score,
                    "passed": r.passed,
                    "details": r.details,
                }
                for name, r in metric_results.items()
            }
        }
        results["samples"].append(sample_result)

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    print(f"\nEvaluation completed in {elapsed:.1f}s ({elapsed/len(samples):.2f}s per sample)")

    # Compute aggregate metrics
    results["aggregate_metrics"] = evaluator.compute_aggregate_scores(all_metric_results)

    # Compute metrics by question type
    by_qtype = {}
    for sample, metrics in zip(results["samples"], all_metric_results):
        qtype = sample["question_type"]
        if qtype not in by_qtype:
            by_qtype[qtype] = []
        by_qtype[qtype].append(metrics)

    for qtype, type_metrics in by_qtype.items():
        results["by_question_type"][qtype] = evaluator.compute_aggregate_scores(type_metrics)

    # Compute metrics by content type
    by_ctype = {}
    for sample, metrics in zip(results["samples"], all_metric_results):
        ctype = sample["content_type"]
        if ctype not in by_ctype:
            by_ctype[ctype] = []
        by_ctype[ctype].append(metrics)

    for ctype, type_metrics in by_ctype.items():
        results["by_content_type"][ctype] = evaluator.compute_aggregate_scores(type_metrics)

    return results


def save_results(results: dict, output_path: Path):
    """Save evaluation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def print_summary(results: dict):
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nModel: {results['model_info']['model_path']}")
    if results['model_info'].get('adapter_path'):
        print(f"Adapter: {results['model_info']['adapter_path']}")

    print(f"\nSamples evaluated: {len(results['samples'])}")
    print(f"Time elapsed: {results.get('elapsed_seconds', 0):.1f}s")

    print("\n--- Aggregate Metrics ---")
    for metric_name, metric_data in sorted(results["aggregate_metrics"].items()):
        mean_score = metric_data.get("mean_score", 0)
        pass_rate = metric_data.get("pass_rate", 0)
        count = metric_data.get("count", 0)
        print(f"  {metric_name:25s}: {mean_score:.3f} (pass rate: {pass_rate:.1%}, n={count})")

    print("\n--- By Question Type (ROUGE-L) ---")
    for qtype, metrics in sorted(results["by_question_type"].items()):
        if "rouge_l" in metrics:
            score = metrics["rouge_l"]["mean_score"]
            count = metrics["rouge_l"]["count"]
            print(f"  {qtype:20s}: {score:.3f} (n={count})")


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM evaluation on baseline or fine-tuned model"
    )

    # Model arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--adapter", "-a",
        type=str,
        default=None,
        help="Path to LoRA adapter directory"
    )

    # Data arguments
    parser.add_argument(
        "--eval-set", "-e",
        type=Path,
        default=Path("eval/eval_sample.jsonl"),
        help="Path to evaluation set JSONL"
    )
    parser.add_argument(
        "--image-base",
        type=str,
        default="training_data",
        help="Base path for image files"
    )

    # Output arguments
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON path (default: eval/reports/<model>_<timestamp>.json)"
    )

    # Evaluation options
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Enable LLM-as-judge metrics (requires ANTHROPIC_API_KEY)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock model for testing evaluation infrastructure"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are an expert BMW E30 M3 service technician. Answer questions about service procedures, specifications, and troubleshooting based on the provided service manual image.",
        help="System prompt for model"
    )

    args = parser.parse_args()

    # Validate eval set exists
    if not args.eval_set.exists():
        print(f"Error: Evaluation set not found: {args.eval_set}")
        print("Run 'make eval-sample' to create the evaluation sample first.")
        sys.exit(1)

    # Generate output path if not specified
    if args.output is None:
        model_name = Path(args.model).name.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_finetuned" if args.adapter else "_baseline"
        args.output = Path(f"eval/reports/{model_name}{suffix}_{timestamp}.json")

    # Create model wrapper
    print("Initializing model wrapper...")
    model_wrapper = create_wrapper(
        model_path=args.model,
        adapter_path=args.adapter,
        mock=args.mock,
    )

    # Create evaluator
    evaluator = VLMEvaluator(use_llm_judge=args.use_llm_judge)

    # Load evaluation set
    print(f"Loading evaluation set from {args.eval_set}")
    eval_set = load_eval_set(args.eval_set)
    print(f"Loaded {len(eval_set)} examples")

    # Run evaluation
    results = run_evaluation(
        model_wrapper=model_wrapper,
        eval_set=eval_set,
        evaluator=evaluator,
        image_base_path=args.image_base,
        use_llm_judge=args.use_llm_judge,
        max_samples=args.max_samples,
        system_prompt=args.system_prompt,
    )

    # Save results
    save_results(results, args.output)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
