#!/usr/bin/env python3
"""
Compare evaluation results between baseline and fine-tuned models.

Generates a detailed markdown report with:
- Overall score comparisons
- Breakdown by question type
- Breakdown by content type
- Safety-critical subset analysis
- Manual probe results
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def load_results(path: Path) -> dict:
    """Load evaluation results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def compute_delta(baseline: float, finetuned: float) -> tuple[float, str]:
    """Compute delta and format as string with indicator."""
    delta = finetuned - baseline
    if delta > 0.01:
        indicator = "+"
    elif delta < -0.01:
        indicator = ""
    else:
        indicator = " "
    return delta, f"{indicator}{delta:+.3f}"


def format_percent_change(baseline: float, finetuned: float) -> str:
    """Format percentage change."""
    if baseline == 0:
        return "N/A"
    pct = ((finetuned - baseline) / baseline) * 100
    return f"{pct:+.1f}%"


def generate_comparison_report(
    baseline: dict,
    finetuned: dict,
    output_path: Path,
) -> str:
    """
    Generate markdown comparison report.

    Returns the markdown content.
    """
    lines = []

    # Header
    lines.append("# VLM Evaluation Comparison Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Model info - support both old format (model_info) and new format (model, adapter_repo)
    base_model = baseline.get('model_info', {}).get('model_path') or baseline.get('model', 'N/A')
    fine_model = finetuned.get('model_info', {}).get('model_path') or finetuned.get('model', 'N/A')
    fine_adapter = (finetuned.get('model_info', {}).get('adapter_path') or
                    finetuned.get('adapter_repo') or
                    finetuned.get('adapter_path') or 'N/A')

    lines.append("## Models Compared")
    lines.append("")
    lines.append("| | Baseline | Fine-tuned |")
    lines.append("|---|---|---|")
    lines.append(f"| Model | `{base_model}` | `{fine_model}` |")
    lines.append(f"| Adapter | - | `{fine_adapter}` |")
    lines.append(f"| Samples | {len(baseline['samples'])} | {len(finetuned['samples'])} |")
    lines.append("")

    # Overall scores
    lines.append("## Overall Scores")
    lines.append("")
    lines.append("| Metric | Baseline | Fine-tuned | Delta | Change |")
    lines.append("|--------|----------|------------|-------|--------|")

    # Get all metrics
    all_metrics = set(baseline["aggregate_metrics"].keys()) | set(finetuned["aggregate_metrics"].keys())

    for metric in sorted(all_metrics):
        base_data = baseline["aggregate_metrics"].get(metric, {})
        fine_data = finetuned["aggregate_metrics"].get(metric, {})

        base_score = base_data.get("mean_score", 0)
        fine_score = fine_data.get("mean_score", 0)

        delta, delta_str = compute_delta(base_score, fine_score)
        pct_change = format_percent_change(base_score, fine_score)

        # Emoji indicator
        if delta > 0.05:
            emoji = ""
        elif delta > 0.01:
            emoji = ""
        elif delta < -0.01:
            emoji = ""
        else:
            emoji = ""

        lines.append(f"| {metric} | {base_score:.3f} | {fine_score:.3f} | {delta_str} {emoji} | {pct_change} |")

    lines.append("")

    # By question type
    lines.append("## By Question Type")
    lines.append("")
    lines.append("ROUGE-L scores by question type:")
    lines.append("")
    lines.append("| Type | Baseline | Fine-tuned | Delta | n (base) |")
    lines.append("|------|----------|------------|-------|----------|")

    all_qtypes = set(baseline["by_question_type"].keys()) | set(finetuned["by_question_type"].keys())

    qtype_data = []
    for qtype in sorted(all_qtypes):
        base_metrics = baseline["by_question_type"].get(qtype, {})
        fine_metrics = finetuned["by_question_type"].get(qtype, {})

        base_rouge = base_metrics.get("rouge_l", {}).get("mean_score", 0)
        fine_rouge = fine_metrics.get("rouge_l", {}).get("mean_score", 0)
        count = base_metrics.get("rouge_l", {}).get("count", 0)

        delta, delta_str = compute_delta(base_rouge, fine_rouge)

        qtype_data.append((qtype, base_rouge, fine_rouge, delta, count))
        lines.append(f"| {qtype} | {base_rouge:.3f} | {fine_rouge:.3f} | {delta_str} | {count} |")

    lines.append("")

    # By content type (if available)
    base_content_types = baseline.get("by_content_type", {})
    fine_content_types = finetuned.get("by_content_type", {})

    if base_content_types or fine_content_types:
        lines.append("## By Content Type")
        lines.append("")
        lines.append("ROUGE-L scores by content type:")
        lines.append("")
        lines.append("| Type | Baseline | Fine-tuned | Delta | n (base) |")
        lines.append("|------|----------|------------|-------|----------|")

        all_ctypes = set(base_content_types.keys()) | set(fine_content_types.keys())

        for ctype in sorted(all_ctypes):
            base_metrics = base_content_types.get(ctype, {})
            fine_metrics = fine_content_types.get(ctype, {})

            base_rouge = base_metrics.get("rouge_l", {}).get("mean_score", 0)
            fine_rouge = fine_metrics.get("rouge_l", {}).get("mean_score", 0)
            count = base_metrics.get("rouge_l", {}).get("count", 0)

            delta, delta_str = compute_delta(base_rouge, fine_rouge)
            lines.append(f"| {ctype} | {base_rouge:.3f} | {fine_rouge:.3f} | {delta_str} | {count} |")

        lines.append("")

    # Safety-critical analysis
    lines.append("## Safety-Critical Analysis")
    lines.append("")

    base_safety = baseline["by_question_type"].get("safety", {})
    fine_safety = finetuned["by_question_type"].get("safety", {})

    if base_safety or fine_safety:
        lines.append("| Metric | Baseline | Fine-tuned | Delta |")
        lines.append("|--------|----------|------------|-------|")

        for metric in ["rouge_l", "numeric_match", "safety_accuracy"]:
            base_score = base_safety.get(metric, {}).get("mean_score", 0)
            fine_score = fine_safety.get(metric, {}).get("mean_score", 0)
            delta, delta_str = compute_delta(base_score, fine_score)
            lines.append(f"| {metric} | {base_score:.3f} | {fine_score:.3f} | {delta_str} |")

        lines.append("")
        lines.append(f"Safety samples: {base_safety.get('rouge_l', {}).get('count', 0)}")
    else:
        lines.append("No safety-tagged samples in evaluation set.")

    lines.append("")

    # Manual probes comparison (if available)
    base_probes = baseline.get("probe_summary", {})
    fine_probes = finetuned.get("probe_summary", {})

    if base_probes or fine_probes:
        lines.append("## Manual Probes")
        lines.append("")
        lines.append("| Metric | Baseline | Fine-tuned | Change |")
        lines.append("|--------|----------|------------|--------|")

        base_rate = base_probes.get("pass_rate", 0)
        fine_rate = fine_probes.get("pass_rate", 0)
        delta = fine_rate - base_rate
        lines.append(f"| Pass Rate | {base_rate:.1%} | {fine_rate:.1%} | {delta:+.1%} |")

        base_crit = base_probes.get("critical_pass_rate", 0)
        fine_crit = fine_probes.get("critical_pass_rate", 0)
        delta_crit = fine_crit - base_crit
        lines.append(f"| Critical Pass Rate | {base_crit:.1%} | {fine_crit:.1%} | {delta_crit:+.1%} |")

        lines.append("")
        lines.append(f"Total probes: {base_probes.get('total', 0)} | Passed: {base_probes.get('passed', 0)} (baseline) vs {fine_probes.get('passed', 0)} (finetuned)")

    lines.append("")

    # Top improvements
    lines.append("## Top Improvements by Question Type")
    lines.append("")

    # Sort by delta
    qtype_data.sort(key=lambda x: x[3], reverse=True)

    lines.append("Categories with largest improvements:")
    lines.append("")
    for qtype, base, fine, delta, count in qtype_data[:5]:
        if delta > 0 and count > 0:
            lines.append(f"- **{qtype}**: +{delta:.3f} (baseline: {base:.3f} -> {fine:.3f})")

    lines.append("")

    # Areas needing attention
    lines.append("## Areas Needing Attention")
    lines.append("")

    qtype_data.sort(key=lambda x: x[3])  # Sort by delta ascending (worst first)
    has_regressions = False

    for qtype, base, fine, delta, count in qtype_data:
        if delta < -0.01 and count > 0:
            has_regressions = True
            lines.append(f"- **{qtype}**: {delta:.3f} (regression from {base:.3f} to {fine:.3f})")

    if not has_regressions:
        lines.append("No significant regressions detected.")

    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    avg_delta = sum(d[3] for d in qtype_data if d[4] > 0) / len([d for d in qtype_data if d[4] > 0])

    if avg_delta > 0.1:
        lines.append("- **Strong improvement** observed. Consider proceeding to deployment testing.")
    elif avg_delta > 0.05:
        lines.append("- **Moderate improvement** observed. Review specific failure cases for further tuning.")
    elif avg_delta > 0:
        lines.append("- **Marginal improvement** observed. Consider additional training data or hyperparameter tuning.")
    else:
        lines.append("- **No improvement** or regression. Review training configuration and data quality.")

    lines.append("")

    # Metadata
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by VLM Evaluation Framework*")

    content = "\n".join(lines)

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    return content


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison report between baseline and fine-tuned model evaluations"
    )
    parser.add_argument(
        "--baseline", "-b",
        type=Path,
        required=True,
        help="Path to baseline evaluation results JSON"
    )
    parser.add_argument(
        "--finetuned", "-f",
        type=Path,
        required=True,
        help="Path to fine-tuned evaluation results JSON"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("eval/reports/comparison.md"),
        help="Output markdown report path"
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading baseline results from {args.baseline}")
    baseline = load_results(args.baseline)

    print(f"Loading fine-tuned results from {args.finetuned}")
    finetuned = load_results(args.finetuned)

    # Generate report
    print(f"Generating comparison report...")
    content = generate_comparison_report(baseline, finetuned, args.output)

    print(f"\nReport saved to {args.output}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)

    all_metrics = set(baseline["aggregate_metrics"].keys()) & set(finetuned["aggregate_metrics"].keys())
    for metric in sorted(all_metrics):
        base = baseline["aggregate_metrics"][metric]["mean_score"]
        fine = finetuned["aggregate_metrics"][metric]["mean_score"]
        delta = fine - base
        indicator = "" if delta > 0.01 else ("" if delta < -0.01 else "")
        print(f"  {metric:25s}: {base:.3f} -> {fine:.3f} ({delta:+.3f}) {indicator}")


if __name__ == "__main__":
    main()
