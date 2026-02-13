#!/usr/bin/env python3
"""
Multi-run evaluation comparison.

Discovers archived eval runs, loads results, and produces a side-by-side
progression report showing how model performance evolves across training
iterations.

Usage:
    # Compare all archived runs
    python eval/compare_runs.py

    # Compare specific runs
    python eval/compare_runs.py --runs run_20260207_134206 run_20260215_120000

    # Custom output path
    python eval/compare_runs.py --output eval/reports/progression.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ARCHIVE_DIR = Path("eval/reports/archive")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_runs(archive_dir: Path, run_names: list[str] | None = None) -> list[Path]:
    """Find run directories, sorted chronologically by directory name."""
    if not archive_dir.is_dir():
        return []

    if run_names:
        dirs = []
        for name in run_names:
            d = archive_dir / name
            if d.is_dir():
                dirs.append(d)
            else:
                print(f"Warning: {d} not found, skipping", file=sys.stderr)
        return dirs

    dirs = sorted(archive_dir.glob("run_*"))
    return [d for d in dirs if d.is_dir()]


def load_run(run_dir: Path) -> dict | None:
    """Load a single run's data: meta, baseline, finetuned."""
    run = {"dir": run_dir, "name": run_dir.name}

    # Load run_meta.json (optional)
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            run["meta"] = json.load(f)
    else:
        run["meta"] = {}

    # Load baseline.json
    baseline_path = run_dir / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            run["baseline"] = json.load(f)
    else:
        run["baseline"] = None

    # Load finetuned.json
    finetuned_path = run_dir / "finetuned.json"
    if finetuned_path.exists():
        with open(finetuned_path) as f:
            run["finetuned"] = json.load(f)
    else:
        run["finetuned"] = None

    # Need at least one result file
    if run["baseline"] is None and run["finetuned"] is None:
        print(f"Warning: {run_dir.name} has no result JSONs, skipping", file=sys.stderr)
        return None

    return run


def get_label(run: dict) -> str:
    """Get human-readable label for a run."""
    return run["meta"].get("label") or run["name"]


def get_timestamp(run: dict) -> str:
    """Get timestamp string for a run."""
    if run["meta"].get("created"):
        return run["meta"]["created"][:19]
    # Try finetuned timestamp, then baseline
    for key in ("finetuned", "baseline"):
        if run.get(key) and run[key].get("timestamp"):
            return run[key]["timestamp"][:19]
    return "-"


def get_adapter(run: dict) -> str:
    """Get adapter identifier for a run."""
    if run["meta"].get("adapter_repo"):
        return run["meta"]["adapter_repo"]
    if run.get("finetuned"):
        return (run["finetuned"].get("adapter_repo")
                or run["finetuned"].get("model_info", {}).get("adapter_path")
                or "-")
    return "-"


def get_sample_count(run: dict) -> int | None:
    """Get evaluation sample count."""
    if run["meta"].get("eval_samples"):
        return run["meta"]["eval_samples"]
    for key in ("finetuned", "baseline"):
        if run.get(key) and run[key].get("samples"):
            return len(run[key]["samples"])
    return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(runs: list[dict]) -> str:
    """Generate multi-run comparison markdown report."""
    lines = []

    lines.append("# Multi-Run Evaluation Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Runs compared: {len(runs)}")
    lines.append("")

    # Use baseline from the earliest run as the reference
    baseline_run = None
    for r in runs:
        if r.get("baseline"):
            baseline_run = r
            break
    baseline = baseline_run["baseline"] if baseline_run else None

    # ------------------------------------------------------------------
    # Runs table
    # ------------------------------------------------------------------
    lines.append("## Runs")
    lines.append("")
    lines.append("| # | Label | Date | Adapter | Samples |")
    lines.append("|---|-------|------|---------|---------|")

    if baseline:
        bl_date = baseline.get("timestamp", "-")[:10]
        bl_samples = len(baseline.get("samples", []))
        lines.append(f"| 0 | baseline | {bl_date} | - | {bl_samples} |")

    for i, run in enumerate(runs, 1):
        label = get_label(run)
        date = get_timestamp(run)[:10]
        adapter = get_adapter(run)
        samples = get_sample_count(run) or "-"
        lines.append(f"| {i} | {label} | {date} | `{adapter}` | {samples} |")

    lines.append("")

    # Collect sample counts for compatibility notes
    sample_counts = set()
    if baseline:
        sample_counts.add(len(baseline.get("samples", [])))
    for run in runs:
        sc = get_sample_count(run)
        if sc:
            sample_counts.add(sc)

    # ------------------------------------------------------------------
    # Overall Metrics
    # ------------------------------------------------------------------
    lines.append("## Overall Metrics")
    lines.append("")

    # Gather all metric names across all runs
    all_metrics = set()
    if baseline:
        all_metrics.update(baseline.get("aggregate_metrics", {}).keys())
    for run in runs:
        if run.get("finetuned"):
            all_metrics.update(run["finetuned"].get("aggregate_metrics", {}).keys())
    metric_names = sorted(all_metrics)

    if metric_names:
        header = "| Run |" + " | ".join(metric_names) + " |"
        sep = "|-----|" + " | ".join(["------"] * len(metric_names)) + " |"
        lines.append(header)
        lines.append(sep)

        # Baseline row
        if baseline:
            bl_agg = baseline.get("aggregate_metrics", {})
            cells = []
            for m in metric_names:
                score = bl_agg.get(m, {}).get("mean_score")
                cells.append(f"{score:.3f}" if score is not None else "-")
            lines.append("| baseline | " + " | ".join(cells) + " |")

        # Each run's finetuned row
        for run in runs:
            label = get_label(run)
            ft = run.get("finetuned")
            if ft:
                ft_agg = ft.get("aggregate_metrics", {})
                cells = []
                for m in metric_names:
                    score = ft_agg.get(m, {}).get("mean_score")
                    cells.append(f"{score:.3f}" if score is not None else "-")
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            else:
                cells = ["-"] * len(metric_names)
                lines.append(f"| {label} | " + " | ".join(cells) + " |")

        lines.append("")

    # ------------------------------------------------------------------
    # Deltas vs Baseline
    # ------------------------------------------------------------------
    if baseline and metric_names:
        lines.append("## Deltas vs Baseline")
        lines.append("")

        header = "| Run |" + " | ".join(metric_names) + " |"
        sep = "|-----|" + " | ".join(["------"] * len(metric_names)) + " |"
        lines.append(header)
        lines.append(sep)

        bl_agg = baseline.get("aggregate_metrics", {})

        for run in runs:
            label = get_label(run)
            ft = run.get("finetuned")
            if ft:
                ft_agg = ft.get("aggregate_metrics", {})
                cells = []
                for m in metric_names:
                    bl_score = bl_agg.get(m, {}).get("mean_score")
                    ft_score = ft_agg.get(m, {}).get("mean_score")
                    if bl_score is not None and ft_score is not None:
                        delta = ft_score - bl_score
                        cells.append(f"{delta:+.3f}")
                    else:
                        cells.append("-")
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            else:
                cells = ["-"] * len(metric_names)
                lines.append(f"| {label} | " + " | ".join(cells) + " |")

        lines.append("")

    # ------------------------------------------------------------------
    # ROUGE-L by Question Type
    # ------------------------------------------------------------------
    lines.append("## ROUGE-L by Question Type")
    lines.append("")

    # Gather union of all question types
    all_qtypes = set()
    if baseline:
        all_qtypes.update(baseline.get("by_question_type", {}).keys())
    for run in runs:
        if run.get("finetuned"):
            all_qtypes.update(run["finetuned"].get("by_question_type", {}).keys())
    qtypes = sorted(all_qtypes)

    if qtypes:
        header = "| Run |" + " | ".join(qtypes) + " |"
        sep = "|-----|" + " | ".join(["------"] * len(qtypes)) + " |"
        lines.append(header)
        lines.append(sep)

        # Baseline row
        if baseline:
            bl_bqt = baseline.get("by_question_type", {})
            cells = []
            for qt in qtypes:
                score = bl_bqt.get(qt, {}).get("rouge_l", {}).get("mean_score")
                cells.append(f"{score:.3f}" if score is not None else "-")
            lines.append("| baseline | " + " | ".join(cells) + " |")

        # Finetuned rows
        for run in runs:
            label = get_label(run)
            ft = run.get("finetuned")
            if ft:
                ft_bqt = ft.get("by_question_type", {})
                cells = []
                for qt in qtypes:
                    score = ft_bqt.get(qt, {}).get("rouge_l", {}).get("mean_score")
                    cells.append(f"{score:.3f}" if score is not None else "-")
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            else:
                cells = ["-"] * len(qtypes)
                lines.append(f"| {label} | " + " | ".join(cells) + " |")

        lines.append("")

    # ------------------------------------------------------------------
    # Manual Probes
    # ------------------------------------------------------------------
    lines.append("## Manual Probes")
    lines.append("")
    lines.append("| Run | Pass Rate | Critical Pass Rate | Passed | Total |")
    lines.append("|-----|-----------|-------------------|--------|-------|")

    if baseline and baseline.get("probe_summary"):
        ps = baseline["probe_summary"]
        lines.append(
            f"| baseline | {ps.get('pass_rate', 0):.1%} | "
            f"{ps.get('critical_pass_rate', 0):.1%} | "
            f"{ps.get('passed', 0)} | {ps.get('total', 0)} |"
        )

    for run in runs:
        label = get_label(run)
        ft = run.get("finetuned")
        if ft and ft.get("probe_summary"):
            ps = ft["probe_summary"]
            lines.append(
                f"| {label} | {ps.get('pass_rate', 0):.1%} | "
                f"{ps.get('critical_pass_rate', 0):.1%} | "
                f"{ps.get('passed', 0)} | {ps.get('total', 0)} |"
            )
        else:
            lines.append(f"| {label} | - | - | - | - |")

    lines.append("")

    # ------------------------------------------------------------------
    # Compatibility Notes
    # ------------------------------------------------------------------
    if len(sample_counts) > 1:
        lines.append("## Compatibility Notes")
        lines.append("")
        lines.append(f"Sample counts differ across runs: {sorted(sample_counts)}. "
                      "Metric comparisons may not be strictly apples-to-apples.")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by VLM Multi-Run Comparison*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-run evaluation comparison report"
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        help="Specific run directory names to compare (default: all)"
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ARCHIVE_DIR,
        help="Archive directory containing run_* dirs"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("eval/reports/multi_run_comparison.md"),
        help="Output markdown report path"
    )

    args = parser.parse_args()

    # Discover runs
    run_dirs = discover_runs(args.archive_dir, args.runs)
    if not run_dirs:
        print("No runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(run_dirs)} run(s):")
    for d in run_dirs:
        print(f"  {d.name}")

    # Load runs
    runs = []
    for d in run_dirs:
        run = load_run(d)
        if run:
            runs.append(run)

    if not runs:
        print("No valid runs to compare.", file=sys.stderr)
        sys.exit(1)

    # Generate report
    report = generate_report(runs)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)

    print(f"\nReport saved to {args.output}")

    # Print quick summary
    print(f"\n{'=' * 60}")
    print("QUICK SUMMARY")
    print(f"{'=' * 60}")

    # Find baseline
    baseline = None
    for r in runs:
        if r.get("baseline"):
            baseline = r["baseline"]
            break

    if baseline:
        bl_agg = baseline.get("aggregate_metrics", {})
        bl_rouge = bl_agg.get("rouge_l", {}).get("mean_score", 0)
        print(f"  Baseline ROUGE-L: {bl_rouge:.3f}")

        for run in runs:
            label = get_label(run)
            ft = run.get("finetuned")
            if ft:
                ft_rouge = ft.get("aggregate_metrics", {}).get("rouge_l", {}).get("mean_score", 0)
                delta = ft_rouge - bl_rouge
                print(f"  {label:30s}: {ft_rouge:.3f} ({delta:+.3f})")


if __name__ == "__main__":
    main()
