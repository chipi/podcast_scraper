"""Human-readable evaluation report generator.

This module generates formatted, human-readable reports from metrics and comparisons.
Reports can be printed to console or saved to files (Markdown, plain text).

This implements RFC-015 Phase 2: Generate human-readable evaluation reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def format_metric_value(value: Optional[float], format_type: str = "float") -> str:
    """Format a metric value for display.

    Args:
        value: The metric value (may be None)
        format_type: Format type ("float", "percentage", "currency", "duration")

    Returns:
        Formatted string representation
    """
    if value is None:
        return "N/A"

    if format_type == "percentage":
        return f"{value * 100:.1f}%"
    elif format_type == "currency":
        return f"${value:.4f}"
    elif format_type == "duration":
        return f"{value:.0f}ms"
    elif format_type == "float":
        return f"{value:.4f}"
    else:
        return str(value)


def format_delta(delta: Optional[float], format_type: str = "float") -> str:
    """Format a delta value for display with sign.

    Args:
        delta: The delta value (may be None)
        format_type: Format type ("float", "percentage", "currency", "duration")

    Returns:
        Formatted string with +/- sign
    """
    if delta is None:
        return "N/A"

    formatted = format_metric_value(abs(delta), format_type)
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{formatted}"


def generate_metrics_report(metrics: Dict[str, Any]) -> str:
    """Generate human-readable metrics report.

    Args:
        metrics: Metrics dictionary from scorer.py

    Returns:
        Formatted report string (Markdown format)
    """
    lines = []
    lines.append("# Experiment Metrics Report")
    lines.append("")
    lines.append(f"**Run ID:** `{metrics.get('run_id', 'unknown')}`")
    lines.append(f"**Dataset ID:** `{metrics.get('dataset_id', 'unknown')}`")
    lines.append(f"**Episode Count:** {metrics.get('episode_count', 0)}")
    lines.append("")

    intrinsic = metrics.get("intrinsic", {})
    if intrinsic:
        lines.append("## Intrinsic Metrics")
        lines.append("")

        # Quality Gates
        gates = intrinsic.get("gates", {})
        if gates:
            lines.append("### Quality Gates")
            lines.append("")
            boilerplate_rate = format_metric_value(gates.get("boilerplate_leak_rate"), "percentage")
            lines.append(f"- **Boilerplate Leak Rate:** {boilerplate_rate}")
            speaker_label_rate = format_metric_value(
                gates.get("speaker_label_leak_rate"), "percentage"
            )
            lines.append(f"- **Speaker Label Leak Rate:** {speaker_label_rate}")
            # Show speaker name leak as warning if available
            warnings = intrinsic.get("warnings", {})
            if warnings.get("speaker_name_leak_rate") is not None:
                speaker_name_rate = format_metric_value(
                    warnings.get("speaker_name_leak_rate"), "percentage"
                )
                lines.append(f"- **Speaker Name Leak Rate (WARN):** {speaker_name_rate}")
            truncation_rate = format_metric_value(gates.get("truncation_rate"), "percentage")
            lines.append(f"- **Truncation Rate:** {truncation_rate}")
            failed = gates.get("failed_episodes", [])
            if failed:
                failed_list = ", ".join(failed[:5])
                failed_suffix = "..." if len(failed) > 5 else ""
                lines.append(
                    f"- **Failed Episodes:** {len(failed)} " f"({failed_list}{failed_suffix})"
                )
                # Show per-episode gate breakdown if available
                episode_failures = gates.get("episode_gate_failures", {})
                if episode_failures:
                    lines.append("")
                    lines.append("**Per-Episode Gate Failures:**")
                    for episode_id in failed:
                        if episode_id in episode_failures:
                            gate_list = ", ".join(episode_failures[episode_id])
                            lines.append(f"- `{episode_id}`: {gate_list}")
            else:
                lines.append("- **Failed Episodes:** None")
            lines.append("")

        # Length Metrics
        length = intrinsic.get("length", {})
        if length:
            lines.append("### Length Metrics")
            lines.append("")
            lines.append(
                f"- **Average Tokens:** {format_metric_value(length.get('avg_tokens'), 'float')}"
            )
            lines.append(
                f"- **Min Tokens:** {format_metric_value(length.get('min_tokens'), 'float')}"
            )
            lines.append(
                f"- **Max Tokens:** {format_metric_value(length.get('max_tokens'), 'float')}"
            )
            lines.append("")

        # Performance Metrics
        performance = intrinsic.get("performance", {})
        if performance:
            lines.append("### Performance Metrics")
            lines.append("")
            avg_latency = format_metric_value(performance.get("avg_latency_ms"), "duration")
            lines.append(f"- **Average Latency:** {avg_latency}")
            lines.append("")

        # Cost Metrics (only for OpenAI runs - ML models skip this section)
        cost = intrinsic.get("cost")
        if cost:
            lines.append("### Cost Metrics")
            lines.append("")
            avg_cost = cost.get("avg_cost_usd")
            total_cost = cost.get("total_cost_usd")
            if avg_cost is not None:
                lines.append(
                    f"- **Average Cost per Episode:** {format_metric_value(avg_cost, 'currency')}"
                )
            if total_cost is not None:
                lines.append(f"- **Total Cost:** {format_metric_value(total_cost, 'currency')}")
            lines.append("")

    # vs_reference Metrics
    vs_reference = metrics.get("vs_reference")
    if vs_reference:
        lines.append("## vs Reference Metrics")
        lines.append("")

        for ref_id, ref_metrics in vs_reference.items():
            if isinstance(ref_metrics, dict) and "error" not in ref_metrics:
                lines.append(f"### vs {ref_id}")
                lines.append("")

                quality = ref_metrics.get("reference_quality")
                if quality:
                    lines.append(f"**Reference Quality:** {quality}")
                    lines.append("")

                # ROUGE scores
                rouge1 = ref_metrics.get("rouge1_f1")
                rouge2 = ref_metrics.get("rouge2_f1")
                rougeL = ref_metrics.get("rougeL_f1")
                if rouge1 is not None or rouge2 is not None or rougeL is not None:
                    lines.append("**ROUGE Scores:**")
                    if rouge1 is not None:
                        lines.append(f"- ROUGE-1 F1: {format_metric_value(rouge1, 'percentage')}")
                    if rouge2 is not None:
                        lines.append(f"- ROUGE-2 F1: {format_metric_value(rouge2, 'percentage')}")
                    if rougeL is not None:
                        lines.append(f"- ROUGE-L F1: {format_metric_value(rougeL, 'percentage')}")
                    lines.append("")

                # BLEU score
                bleu = ref_metrics.get("bleu")
                if bleu is not None:
                    lines.append(f"**BLEU Score:** {format_metric_value(bleu, 'percentage')}")
                    lines.append("")

                # WER score
                wer = ref_metrics.get("wer")
                if wer is not None:
                    lines.append(
                        f"**Word Error Rate (WER):** {format_metric_value(wer, 'percentage')}"
                    )
                    lines.append("")

                # Embedding similarity
                embedding = ref_metrics.get("embedding_cosine")
                if embedding is not None:
                    embedding_sim = format_metric_value(embedding, "percentage")
                    lines.append(f"**Embedding Cosine Similarity:** {embedding_sim}")
                    lines.append("")
            elif isinstance(ref_metrics, dict) and "error" in ref_metrics:
                lines.append(f"### vs {ref_id}")
                lines.append("")
                lines.append(f"**Error:** {ref_metrics['error']}")
                lines.append("")

    return "\n".join(lines)


def generate_comparison_report(
    comparison: Dict[str, Any], baseline_metrics: Optional[Dict[str, Any]] = None
) -> str:
    """Generate human-readable comparison report.

    Args:
        comparison: Comparison dictionary from comparator.py
        baseline_metrics: Optional baseline metrics for context

    Returns:
        Formatted report string (Markdown format)
    """
    lines = []
    lines.append("# Baseline Comparison Report")
    lines.append("")
    lines.append(f"**Baseline ID:** `{comparison.get('baseline_id', 'unknown')}`")
    lines.append(f"**Experiment Run ID:** `{comparison.get('experiment_run_id', 'unknown')}`")
    lines.append(f"**Dataset ID:** `{comparison.get('dataset_id', 'unknown')}`")
    lines.append("")

    deltas = comparison.get("deltas", {})
    if not deltas:
        lines.append("No deltas computed (metrics may be missing).")
        return "\n".join(lines)

    lines.append("## Deltas (Experiment - Baseline)")
    lines.append("")

    # Cost deltas
    if "cost_total_usd" in deltas:
        delta = deltas["cost_total_usd"]
        lines.append("### Cost")
        lines.append(f"- **Total Cost Delta:** {format_delta(delta, 'currency')}")
        if delta > 0:
            lines.append("  ⚠️  Cost increased")
        elif delta < 0:
            lines.append("  ✅ Cost decreased")
        lines.append("")

    # Performance deltas
    if "avg_latency_ms" in deltas:
        delta = deltas["avg_latency_ms"]
        lines.append("### Performance")
        lines.append(f"- **Average Latency Delta:** {format_delta(delta, 'duration')}")
        if delta > 0:
            lines.append("  ⚠️  Latency increased")
        elif delta < 0:
            lines.append("  ✅ Latency decreased")
        lines.append("")

    # Gate regressions
    gate_regressions = deltas.get("gate_regressions", [])
    if gate_regressions:
        lines.append("### Quality Gate Regressions")
        lines.append("")
        lines.append("⚠️  **WARNING:** The following quality gates regressed:")
        for gate in gate_regressions:
            lines.append(f"- {gate}")
        lines.append("")
    else:
        lines.append("### Quality Gates")
        lines.append("")
        lines.append("✅ No gate regressions detected")
        lines.append("")

    # vs_reference deltas
    vs_ref_deltas = {k: v for k, v in deltas.items() if k.startswith("rougeL_f1_vs_")}
    if vs_ref_deltas:
        lines.append("### vs Reference Deltas")
        lines.append("")
        for metric_key, delta in vs_ref_deltas.items():
            ref_id = metric_key.replace("rougeL_f1_vs_", "")
            lines.append(f"- **ROUGE-L F1 vs {ref_id}:** {format_delta(delta, 'percentage')}")
            if delta > 0:
                lines.append("  ✅ Quality improved")
            elif delta < 0:
                lines.append("  ⚠️  Quality regressed")
        lines.append("")

    return "\n".join(lines)


def save_report(report: str, output_path: Path, format: str = "markdown") -> None:
    """Save report to file.

    Args:
        report: Report content string
        output_path: Path to save report
        format: Report format ("markdown" or "txt")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to: {output_path}")


def print_report(report: str) -> None:
    """Print report to console.

    Args:
        report: Report content string
    """
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80 + "\n")
