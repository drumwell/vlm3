#!/usr/bin/env python3
"""
Modal evaluation app for Qwen2-VL baseline and fine-tuned model comparison.

Runs VLM evaluation on Modal's GPUs without requiring local GPU access.

Prerequisites:
    1. Modal account: https://modal.com
    2. Modal CLI: pip install modal && modal setup
    3. HuggingFace secret: modal secret create huggingface HF_TOKEN=your_token
    4. Eval sample created: make eval-sample

Usage:
    # Run baseline evaluation
    modal run eval/modal_eval.py

    # Run fine-tuned evaluation (adapter from HuggingFace)
    modal run eval/modal_eval.py --adapter-repo username/vlm3-lora

    # Run fine-tuned evaluation (adapter from Modal volume)
    modal run eval/modal_eval.py --adapter-path /checkpoints/vlm3-lora/final

    # Quick test (10 samples)
    modal run eval/modal_eval.py --max-samples 10

    # Run manual probes only
    modal run eval/modal_eval.py --probes-only
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import modal

# Modal configuration
GPU_TYPE = "A100"  # 40GB is enough for inference
TIMEOUT_HOURS = 2
VOLUME_NAME = "vlm3-checkpoints"

# Default paths
DEFAULT_DATASET_REPO = "drumwell/vlm3"
DEFAULT_EVAL_SAMPLE = "eval/eval_sample.jsonl"
DEFAULT_PROBES = "eval/benchmarks/manual_probes.json"

# Docker image with evaluation dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.45.0",
        "accelerate>=0.27.0",
        "peft>=0.10.0",
        "qwen-vl-utils>=0.0.8",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "huggingface_hub>=0.21.0",
        "rouge-score>=0.1.2",
        "nltk>=3.8.0",
    )
)

# Create Modal app
app = modal.App("vlm3-eval")

# Persistent volume for checkpoints (shared with training)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# =============================================================================
# Metrics (simplified versions for Modal - no deepeval dependency)
# =============================================================================

class SimpleMetrics:
    """Simplified metrics that run without external dependencies."""

    @staticmethod
    def rouge_l(prediction: str, reference: str) -> float:
        """Compute ROUGE-L F1 score."""
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure

    @staticmethod
    def extract_numbers(text: str) -> list[tuple[float, str]]:
        """Extract numbers with units from text."""
        pattern = r'(\d+(?:\.\d+)?)\s*(nm|mm|bar|l|°c)?'
        matches = re.findall(pattern, text.lower())
        return [(float(m[0]), m[1]) for m in matches]

    @staticmethod
    def numeric_match(prediction: str, reference: str) -> float:
        """Check if numeric values match within tolerance."""
        pred_nums = SimpleMetrics.extract_numbers(prediction)
        ref_nums = SimpleMetrics.extract_numbers(reference)

        if not ref_nums:
            return 1.0

        tolerances = {"nm": 1.0, "mm": 0.05, "bar": 0.1, "l": 0.1, "°c": 1.0}
        matches = 0

        for ref_val, ref_unit in ref_nums:
            tol = tolerances.get(ref_unit, 0.01 * ref_val)
            for pred_val, _ in pred_nums:
                if abs(pred_val - ref_val) <= tol:
                    matches += 1
                    break

        return matches / len(ref_nums)

    @staticmethod
    def keyword_presence(prediction: str, keywords: list[str]) -> float:
        """Check fraction of keywords present."""
        if not keywords:
            return 1.0
        pred_lower = prediction.lower()
        found = sum(1 for kw in keywords if kw.lower() in pred_lower)
        return found / len(keywords)

    @staticmethod
    def unit_consistency(prediction: str) -> float:
        """Check for non-canonical units."""
        non_canonical = [
            (r'\d+\s*(?:ft[\s-]?lb|foot[\s-]?pound)', "ft-lbs"),
            (r'\d+\s*psi', "psi"),
            (r'\d+(?:\.\d+)?\s*(?:inch|in\.|")', "inches"),
        ]
        issues = sum(1 for pattern, _ in non_canonical if re.search(pattern, prediction, re.IGNORECASE))
        return max(0, 1.0 - issues * 0.25)


# =============================================================================
# Evaluation Functions
# =============================================================================

def extract_qa(record: dict) -> tuple[str, str]:
    """Extract question and answer from conversation record."""
    conversations = record.get("conversations", [])
    question, answer = "", ""

    for turn in conversations:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            if isinstance(content, str):
                question = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        question = item.get("text", "")
        elif role == "assistant":
            answer = content

    return question, answer


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_HOURS * 3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/checkpoints": volume},
)
def run_evaluation(
    dataset_repo: str,
    eval_records: list[dict],
    probes: list[dict],
    adapter_repo: str | None = None,
    adapter_path: str | None = None,
    max_samples: int | None = None,
) -> dict:
    """
    Run VLM evaluation on Modal GPU.

    Args:
        dataset_repo: HuggingFace dataset repo for images
        eval_records: List of evaluation records (from eval_sample.jsonl)
        probes: List of manual probe test cases
        adapter_repo: HuggingFace repo containing LoRA adapter
        adapter_path: Local path to LoRA adapter (in Modal volume)
        max_samples: Limit number of samples

    Returns:
        Evaluation results dict
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    from PIL import Image
    from tqdm import tqdm
    from huggingface_hub import hf_hub_download
    import tempfile

    hf_token = os.environ.get("HF_TOKEN")
    metrics = SimpleMetrics()

    # Setup image cache
    cache_dir = Path(tempfile.mkdtemp(prefix="vlm3_eval_"))

    def download_image(image_path: str) -> Image.Image | None:
        try:
            local_path = hf_hub_download(
                repo_id=dataset_repo,
                filename=image_path,
                repo_type="dataset",
                token=hf_token,
                cache_dir=str(cache_dir),
            )
            return Image.open(local_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            return None

    # Determine if this is baseline or fine-tuned
    is_finetuned = adapter_repo is not None or adapter_path is not None
    eval_type = "finetuned" if is_finetuned else "baseline"

    print("=" * 60)
    print(f"VLM3 Evaluation - {eval_type.upper()}")
    print("=" * 60)
    print(f"Dataset: {dataset_repo}")
    print(f"Adapter repo: {adapter_repo or 'None'}")
    print(f"Adapter path: {adapter_path or 'None'}")
    print(f"Eval samples: {len(eval_records)}")
    print(f"Manual probes: {len(probes)}")
    print(f"Max samples: {max_samples or 'all'}")
    print("=" * 60)

    # Load model
    print("\n🤖 Loading Qwen2-VL-7B-Instruct...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True,
    )

    # Load adapter if specified
    if adapter_repo:
        print(f"\n🔧 Loading LoRA adapter from HuggingFace: {adapter_repo}")
        model = PeftModel.from_pretrained(model, adapter_repo, token=hf_token)
    elif adapter_path:
        print(f"\n🔧 Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print("Model loaded successfully")

    # Apply max_samples limit
    if max_samples and eval_records:
        eval_records = eval_records[:max_samples]

    # Helper function to generate response
    def generate_response(image: Image.Image, question: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
            )

        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return response.strip()

    results = {
        "eval_type": eval_type,
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "adapter_repo": adapter_repo,
        "adapter_path": adapter_path,
        "timestamp": datetime.now().isoformat(),
        "samples": [],
        "probes": [],
        "aggregate_metrics": {},
        "by_question_type": {},
    }

    # Run evaluation on eval sample
    if eval_records:
        print(f"\n🔬 Evaluating {len(eval_records)} samples...")
        all_scores = {"rouge_l": [], "numeric": [], "unit": [], "keyword": []}
        by_qtype = {}

        for i, record in enumerate(tqdm(eval_records, desc="Eval samples")):
            question, reference = extract_qa(record)
            image_path = record.get("image", "")
            metadata = record.get("metadata", {})
            qtype = metadata.get("question_type", "unknown")

            image = download_image(image_path)
            if image is None:
                continue

            try:
                prediction = generate_response(image, question)
            except Exception as e:
                print(f"\nError on sample {i}: {e}")
                prediction = f"[ERROR: {str(e)}]"

            # Compute metrics
            rouge = metrics.rouge_l(prediction, reference)
            numeric = metrics.numeric_match(prediction, reference)
            unit = metrics.unit_consistency(prediction)
            keywords = metadata.get("keywords", [])
            kw_score = metrics.keyword_presence(prediction, keywords) if keywords else 1.0

            all_scores["rouge_l"].append(rouge)
            all_scores["numeric"].append(numeric)
            all_scores["unit"].append(unit)
            all_scores["keyword"].append(kw_score)

            if qtype not in by_qtype:
                by_qtype[qtype] = {"rouge_l": [], "count": 0}
            by_qtype[qtype]["rouge_l"].append(rouge)
            by_qtype[qtype]["count"] += 1

            results["samples"].append({
                "index": i,
                "question": question[:200],
                "reference": reference[:200],
                "prediction": prediction[:200],
                "question_type": qtype,
                "metrics": {
                    "rouge_l": rouge,
                    "numeric_match": numeric,
                    "unit_consistency": unit,
                    "keyword_presence": kw_score,
                }
            })

        # Compute aggregates
        for metric_name, scores in all_scores.items():
            if scores:
                results["aggregate_metrics"][metric_name] = {
                    "mean_score": sum(scores) / len(scores),
                    "count": len(scores),
                }

        for qtype, data in by_qtype.items():
            scores = data["rouge_l"]
            results["by_question_type"][qtype] = {
                "rouge_l": {
                    "mean_score": sum(scores) / len(scores) if scores else 0,
                    "count": data["count"],
                }
            }

    # Run manual probes
    if probes:
        print(f"\n🎯 Running {len(probes)} manual probes...")

        for probe in tqdm(probes, desc="Manual probes"):
            image_path = probe.get("image", "")
            question = probe.get("question", "")

            image = download_image(image_path)
            if image is None:
                results["probes"].append({
                    "id": probe.get("id"),
                    "status": "skipped",
                    "reason": "image not found",
                })
                continue

            try:
                prediction = generate_response(image, question)
            except Exception as e:
                results["probes"].append({
                    "id": probe.get("id"),
                    "status": "error",
                    "reason": str(e),
                })
                continue

            # Evaluate probe
            expected = probe.get("expected", "")
            keywords = probe.get("keywords", probe.get("expected_keywords", []))
            is_critical = probe.get("is_critical", False)
            category = probe.get("category", "")

            # For hallucination/OOD probes, check expected_behavior
            expected_behavior = probe.get("expected_behavior", "")

            if expected:
                rouge = metrics.rouge_l(prediction, expected)
                passed = rouge > 0.5
            elif keywords:
                kw_score = metrics.keyword_presence(prediction, keywords)
                passed = kw_score >= 0.5
            elif expected_behavior:
                # For hallucination probes - check if model appropriately declines
                decline_phrases = ["not shown", "not visible", "cannot determine", "not present",
                                   "doesn't show", "does not show", "not in this", "different",
                                   "does not contain", "doesn't contain", "no information"]
                has_decline = any(phrase in prediction.lower() for phrase in decline_phrases)
                passed = has_decline if "decline" in expected_behavior.lower() else True
            else:
                passed = len(prediction) > 10  # Basic sanity check

            results["probes"].append({
                "id": probe.get("id"),
                "category": category,
                "question": question[:100],
                "prediction": prediction[:200],
                "is_critical": is_critical,
                "passed": passed,
            })

        # Compute probe pass rate
        probe_results = results["probes"]
        passed = sum(1 for p in probe_results if p.get("passed", False))
        critical_probes = [p for p in probe_results if p.get("is_critical", False)]
        critical_passed = sum(1 for p in critical_probes if p.get("passed", False))

        results["probe_summary"] = {
            "total": len(probe_results),
            "passed": passed,
            "pass_rate": passed / len(probe_results) if probe_results else 0,
            "critical_total": len(critical_probes),
            "critical_passed": critical_passed,
            "critical_pass_rate": critical_passed / len(critical_probes) if critical_probes else 0,
        }

    print("\n✅ Evaluation complete!")
    return results


@app.local_entrypoint()
def main(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    adapter_repo: str = None,
    adapter_path: str = None,
    max_samples: int = None,
    probes_only: bool = False,
    skip_probes: bool = False,
    output: str = None,
):
    """
    Run VLM evaluation on Modal.

    Examples:
        # Baseline evaluation
        modal run eval/modal_eval.py

        # Fine-tuned evaluation (adapter from HuggingFace)
        modal run eval/modal_eval.py --adapter-repo drumwell/vlm3-lora

        # Fine-tuned evaluation (adapter from Modal volume)
        modal run eval/modal_eval.py --adapter-path /checkpoints/vlm3-lora/final

        # Quick test
        modal run eval/modal_eval.py --max-samples 10

        # Only manual probes
        modal run eval/modal_eval.py --probes-only
    """
    print("Starting evaluation on Modal...")

    # Load local files
    eval_sample_path = Path(DEFAULT_EVAL_SAMPLE)
    probes_path = Path(DEFAULT_PROBES)

    # Load eval sample
    eval_records = []
    if not probes_only:
        if not eval_sample_path.exists():
            print(f"Error: Eval sample not found at {eval_sample_path}")
            print("Run 'make eval-sample' first to create the stratified sample.")
            return
        with open(eval_sample_path, "r") as f:
            for line in f:
                if line.strip():
                    eval_records.append(json.loads(line))
        print(f"Loaded {len(eval_records)} eval samples from {eval_sample_path}")

    # Load probes
    probes = []
    if not skip_probes:
        if probes_path.exists():
            with open(probes_path) as f:
                probes = json.load(f)
            print(f"Loaded {len(probes)} manual probes from {probes_path}")
        else:
            print(f"Warning: Manual probes not found at {probes_path}")

    # Run evaluation on Modal
    print(f"\nSending {len(eval_records)} eval samples and {len(probes)} probes to Modal...")
    results = run_evaluation.remote(
        dataset_repo=dataset_repo,
        eval_records=eval_records,
        probes=probes,
        adapter_repo=adapter_repo,
        adapter_path=adapter_path,
        max_samples=max_samples,
    )

    # Determine output path
    if output is None:
        eval_type = "finetuned" if (adapter_repo or adapter_path) else "baseline"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"eval/reports/{eval_type}_{timestamp}.json"

    # Save results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Type: {results['eval_type']}")
    print(f"Samples evaluated: {len(results['samples'])}")

    if results.get("aggregate_metrics"):
        print("\nAggregate Metrics:")
        for metric, data in results["aggregate_metrics"].items():
            print(f"  {metric}: {data['mean_score']:.3f} (n={data['count']})")

    if results.get("probe_summary"):
        ps = results["probe_summary"]
        print(f"\nManual Probes:")
        print(f"  Pass rate: {ps['passed']}/{ps['total']} ({ps['pass_rate']:.1%})")
        print(f"  Critical: {ps['critical_passed']}/{ps['critical_total']} ({ps['critical_pass_rate']:.1%})")

    print("\n" + "=" * 60)
