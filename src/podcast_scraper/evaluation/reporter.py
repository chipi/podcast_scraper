"""Human-readable evaluation report generator.

This module generates formatted, human-readable reports from metrics and comparisons.
Reports can be printed to console or saved to files (Markdown, plain text).

This implements RFC-015 Phase 2: Generate human-readable evaluation reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def generate_metrics_report(metrics: Dict[str, Any]) -> str:  # noqa: C901
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
            med = performance.get("median_latency_ms")
            if med is not None:
                lines.append(f"- **Median Latency:** {format_metric_value(med, 'duration')}")
            p95 = performance.get("p95_latency_ms")
            if p95 is not None:
                lines.append(f"- **P95 Latency:** {format_metric_value(p95, 'duration')}")
            steady = performance.get("avg_latency_ms_excluding_first")
            if steady is not None:
                lines.append(
                    "- **Avg Latency (excl. first episode):** "
                    f"{format_metric_value(steady, 'duration')}"
                )
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

    # vs_reference deltas (ROUGE, BLEU, embedding, coverage - value metrics)
    vs_ref_deltas = {k: v for k, v in deltas.items() if "_vs_" in k and k != "gate_regressions"}
    percentage_metrics = ("rouge1_f1", "rouge2_f1", "rougeL_f1", "bleu", "wer", "embedding_cosine")
    if vs_ref_deltas:
        lines.append("### vs Reference (Value) Deltas")
        lines.append("")
        # Group by ref_id for readability
        by_ref: Dict[str, List[tuple]] = {}
        for metric_key, delta in vs_ref_deltas.items():
            if delta is None:
                continue
            # key is like "rouge1_f1_vs_silver_gpt4o_benchmark_v1"
            parts = metric_key.rsplit("_vs_", 1)
            if len(parts) != 2:
                continue
            metric_name, ref_id = parts[0], parts[1]
            by_ref.setdefault(ref_id, []).append((metric_name, delta))
        for ref_id in sorted(by_ref.keys()):
            lines.append(f"**Reference: {ref_id}**")
            for metric_name, delta in sorted(by_ref[ref_id]):
                fmt = "percentage" if metric_name in percentage_metrics else "float"
                direction = (
                    "✅ improved" if delta > 0 else "⚠️ regressed" if delta < 0 else "unchanged"
                )
                lines.append(f"- {metric_name}: {format_delta(delta, fmt)} ({direction})")
            lines.append("")
        lines.append("")

    # Quality uplift interpretation: hybrid pipeline vs ML prod baseline
    baseline_id = comparison.get("baseline_id", "")
    experiment_id = comparison.get("experiment_run_id", "")
    if baseline_id == "baseline_ml_prod_authority_v1" and "hybrid" in experiment_id.lower():
        lines.append("## Quality Uplift: Hybrid vs Prod")
        lines.append("")
        lines.append(
            "This experiment compares the **hybrid MAP-REDUCE pipeline** (LongT5 MAP + "
            "FLAN-T5 or other REDUCE via transformers) to **production ML** "
            "(Pegasus-CNN MAP + LED-base REDUCE)."
        )
        lines.append("")
        rouge_deltas = [
            v
            for k, v in vs_ref_deltas.items()
            if v is not None
            and any(k.startswith(r) for r in ("rouge1_f1_", "rouge2_f1_", "rougeL_f1_"))
        ]
        if rouge_deltas:
            avg_rouge_delta = sum(rouge_deltas) / len(rouge_deltas)
            if avg_rouge_delta > 0:
                lines.append(
                    f"- **Summary quality (ROUGE):** Improved on average "
                    f"(delta +{avg_rouge_delta:.2%}). The new hybrid stack and transformers "
                    "models are delivering better overlap with reference summaries."
                )
            elif avg_rouge_delta < 0:
                lines.append(
                    f"- **Summary quality (ROUGE):** Decreased on average "
                    f"(delta {avg_rouge_delta:.2%}). Check latency/cost trade-offs and "
                    "per-episode metrics before concluding."
                )
            else:
                lines.append("- **Summary quality (ROUGE):** No material change vs reference(s).")
        lat_delta = deltas.get("avg_latency_ms")
        if lat_delta is not None:
            if lat_delta > 0:
                lines.append(
                    f"- **Latency:** Hybrid run is slower by ~{lat_delta / 1000:.1f}s per episode "
                    "on average; acceptable if quality uplift justifies it."
                )
            else:
                lines.append(
                    f"- **Latency:** Hybrid run is faster by ~{-lat_delta / 1000:.1f}s per episode."
                )
        lines.append("")
        lines.append(
            "Use this comparison to decide whether to promote the hybrid config to prod "
            "or to iterate on model/params (e.g. REDUCE model, beam size, length penalty)."
        )
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
