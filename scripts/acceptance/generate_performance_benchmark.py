#!/usr/bin/env python3
"""Generate performance benchmarking report from acceptance test results.

This script analyzes acceptance test runs and generates performance benchmarking
reports grouped by provider/model configurations, enabling easy comparison of
different LLM providers and models.

Usage:
    python scripts/acceptance/generate_performance_benchmark.py \
        --session-id session_20260206_103000 \
        --output-dir .test_outputs/acceptance \
        [--output-format markdown|json|both] \
        [--compare-baseline baseline_id]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


def load_baseline(baseline_id: str, output_dir: Path) -> Dict[str, Any]:
    """Load baseline data.

    Args:
        baseline_id: Baseline identifier
        output_dir: Output directory

    Returns:
        Baseline data dict
    """
    baseline_path = output_dir / "baselines" / baseline_id / "baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    with open(baseline_path, "r") as f:
        return json.load(f)


def load_session_data(session_id: str, output_dir: Path) -> Dict[str, Any]:
    """Load session data from JSON file.

    Args:
        session_id: Session identifier (e.g., '20260208_093757' or 'session_20260208_093757')
        output_dir: Output directory

    Returns:
        Session data dict
    """
    # Normalize session_id (remove 'session_' prefix if present)
    if session_id.startswith("session_"):
        session_id = session_id.replace("session_", "", 1)

    # Try new structure first: sessions/session_{id}/session.json
    session_path = output_dir / "sessions" / f"session_{session_id}" / "session.json"
    if not session_path.exists():
        # Fallback to old structure: session_{id}.json (for backwards compatibility)
        session_path = output_dir / f"session_{session_id}.json"
        if not session_path.exists():
            raise FileNotFoundError(
                f"Session file not found. Tried:\n"
                f"  - {output_dir / 'sessions' / f'session_{session_id}' / 'session.json'}\n"
                f"  - {output_dir / f'session_{session_id}.json'}"
            )

    with open(session_path, "r") as f:
        return json.load(f)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate statistical metrics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dict with statistical metrics
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std_dev": 0.0,
        }

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def group_runs_by_provider(runs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group runs by provider/model configuration.

    Args:
        runs: List of run data dicts

    Returns:
        Dict mapping provider keys to lists of runs
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for run in runs:
        provider_info = run.get("provider_info", {})
        if not provider_info:
            # Skip runs without provider info
            continue

        # Create a key that identifies the provider/model combination
        # Format: "speaker:{provider}:{model}_summary:{provider}:{model}"
        speaker_provider = provider_info.get("speaker_provider", "unknown")
        speaker_model = provider_info.get("speaker_model", "unknown")
        summary_provider = provider_info.get("summary_provider", "unknown")
        summary_model = provider_info.get("summary_model", "unknown")

        # For transformers, use map_model
        if summary_provider in ("transformers", "local"):
            summary_model = provider_info.get("summary_map_model", "unknown")

        key = (
            f"speaker:{speaker_provider}:{speaker_model}_summary:{summary_provider}:{summary_model}"
        )
        groups[key].append(run)

    return dict(groups)


def generate_provider_key(provider_info: Dict[str, Any]) -> str:
    """Generate a human-readable provider key.

    Args:
        provider_info: Provider info dict

    Returns:
        Human-readable key string
    """
    speaker_provider = provider_info.get("speaker_provider", "unknown")
    speaker_model = provider_info.get("speaker_model", "unknown")
    summary_provider = provider_info.get("summary_provider", "unknown")
    summary_model = provider_info.get("summary_model", "unknown")

    # For transformers, use map_model
    if summary_provider in ("transformers", "local"):
        summary_model = provider_info.get("summary_map_model", "unknown")

    return f"{speaker_provider}:{speaker_model} + {summary_provider}:{summary_model}"


def analyze_provider_group(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze a group of runs with the same provider/model configuration.

    Args:
        runs: List of run data dicts (same provider/model)

    Returns:
        Analysis dict with aggregated metrics
    """
    if not runs:
        return {}

    # Filter out dry-run runs
    non_dry_runs = [r for r in runs if not r.get("is_dry_run", False)]
    if not non_dry_runs:
        return {"note": "All runs were dry-run mode"}

    # Extract metrics
    durations = [r.get("duration_seconds", 0) for r in non_dry_runs]
    episodes = [r.get("episodes_processed", 0) for r in non_dry_runs]
    memory_values = [
        r.get("resource_usage", {}).get("peak_memory_mb", 0)
        for r in non_dry_runs
        if r.get("resource_usage", {}).get("peak_memory_mb") is not None
    ]
    cpu_times = [
        r.get("resource_usage", {}).get("cpu_time_seconds", 0)
        for r in non_dry_runs
        if r.get("resource_usage", {}).get("cpu_time_seconds") is not None
    ]

    # Calculate per-episode metrics
    total_episodes = sum(episodes)
    total_duration = sum(durations)
    seconds_per_episode = total_duration / max(1, total_episodes) if total_episodes > 0 else 0
    episodes_per_second = total_episodes / max(0.1, total_duration) if total_duration > 0 else 0

    # Memory per episode
    avg_memory = statistics.mean(memory_values) if memory_values else None
    memory_per_episode = (
        avg_memory / max(1, total_episodes / len(non_dry_runs))
        if avg_memory and total_episodes > 0
        else None
    )

    # Get provider info from first run
    provider_info = non_dry_runs[0].get("provider_info", {})

    return {
        "provider_info": provider_info,
        "run_count": len(non_dry_runs),
        "total_episodes": total_episodes,
        "total_duration_seconds": total_duration,
        "duration_stats": calculate_statistics(durations),
        "episodes_stats": calculate_statistics(episodes),
        "seconds_per_episode": round(seconds_per_episode, 2),
        "episodes_per_second": round(episodes_per_second, 3),
        "memory_stats": calculate_statistics(memory_values) if memory_values else None,
        "memory_per_episode_mb": round(memory_per_episode, 2) if memory_per_episode else None,
        "cpu_time_stats": calculate_statistics(cpu_times) if cpu_times else None,
        "success_rate": sum(1 for r in non_dry_runs if r.get("exit_code", 1) == 0)
        / len(non_dry_runs)
        * 100,
    }


def compare_provider_groups_with_baseline(
    current_groups: Dict[str, List[Dict[str, Any]]],
    baseline_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare current provider groups with baseline.

    Args:
        current_groups: Current runs grouped by provider/model
        baseline_data: Baseline data

    Returns:
        Comparison results grouped by provider/model
    """
    baseline_runs = baseline_data.get("runs", [])
    baseline_groups = group_runs_by_provider(baseline_runs)

    comparisons = {}
    for key, current_runs in current_groups.items():
        baseline_runs_for_key = baseline_groups.get(key, [])

        if not baseline_runs_for_key:
            comparisons[key] = {
                "status": "no_baseline",
                "message": "No baseline found for this provider/model configuration",
            }
            continue

        # Analyze both current and baseline groups
        current_analysis = analyze_provider_group(current_runs)
        baseline_analysis = analyze_provider_group(baseline_runs_for_key)

        if not current_analysis or "note" in current_analysis:
            continue
        if not baseline_analysis or "note" in baseline_analysis:
            comparisons[key] = {
                "status": "no_baseline",
                "message": "Baseline data incomplete for this configuration",
            }
            continue

        # Compare metrics
        duration_delta = (
            current_analysis["seconds_per_episode"] - baseline_analysis["seconds_per_episode"]
        )
        duration_percent_change = (
            duration_delta / max(0.1, baseline_analysis["seconds_per_episode"]) * 100
        )

        throughput_delta = (
            current_analysis["episodes_per_second"] - baseline_analysis["episodes_per_second"]
        )
        throughput_percent_change = (
            throughput_delta / max(0.001, baseline_analysis["episodes_per_second"]) * 100
        )

        memory_delta = None
        memory_percent_change = None
        if (
            current_analysis.get("memory_per_episode_mb") is not None
            and baseline_analysis.get("memory_per_episode_mb") is not None
        ):
            memory_delta = (
                current_analysis["memory_per_episode_mb"]
                - baseline_analysis["memory_per_episode_mb"]
            )
            memory_percent_change = (
                memory_delta / max(1, baseline_analysis["memory_per_episode_mb"]) * 100
            )

        # Determine regression/improvement
        is_regression = False
        is_improvement = False
        change_reasons = []

        if duration_percent_change > 20:  # 20% slower
            is_regression = True
            change_reasons.append(f"Time per episode increased by {duration_percent_change:.1f}%")
        elif duration_percent_change < -10:  # 10% faster
            is_improvement = True
            change_reasons.append(
                f"Time per episode decreased by {abs(duration_percent_change):.1f}%"
            )

        if throughput_percent_change < -20:  # 20% slower throughput
            is_regression = True
            change_reasons.append(f"Throughput decreased by {abs(throughput_percent_change):.1f}%")
        elif throughput_percent_change > 10:  # 10% faster throughput
            is_improvement = True
            change_reasons.append(f"Throughput increased by {throughput_percent_change:.1f}%")

        if memory_delta is not None and memory_delta > 100:  # 100MB more memory
            is_regression = True
            change_reasons.append(f"Memory per episode increased by {memory_delta:.0f}MB")
        elif memory_delta is not None and memory_delta < -50:  # 50MB less memory
            is_improvement = True
            change_reasons.append(f"Memory per episode decreased by {abs(memory_delta):.0f}MB")

        status = "regression" if is_regression else ("improvement" if is_improvement else "ok")

        comparisons[key] = {
            "status": status,
            "current": current_analysis,
            "baseline": baseline_analysis,
            "duration_delta_seconds": round(duration_delta, 2),
            "duration_percent_change": round(duration_percent_change, 2),
            "throughput_delta": round(throughput_delta, 3),
            "throughput_percent_change": round(throughput_percent_change, 2),
            "memory_delta_mb": round(memory_delta, 2) if memory_delta is not None else None,
            "memory_percent_change": (
                round(memory_percent_change, 2) if memory_percent_change is not None else None
            ),
            "change_reasons": change_reasons,
        }

    return {
        "baseline_id": baseline_data.get("baseline_id"),
        "total_comparisons": len(comparisons),
        "regressions": sum(1 for c in comparisons.values() if c.get("status") == "regression"),
        "improvements": sum(1 for c in comparisons.values() if c.get("status") == "improvement"),
        "comparisons": comparisons,
    }


def generate_benchmark_report(  # noqa: C901
    session_data: Dict[str, Any],
    baseline_comparison: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate performance benchmarking report.

    Args:
        session_data: Session data

    Returns:
        Markdown report string
    """
    runs = session_data.get("runs", [])
    if not runs:
        return "# Performance Benchmarking Report\n\nNo runs found in session data.\n"

    # Group runs by provider/model
    provider_groups = group_runs_by_provider(runs)

    report = []
    report.append("# Performance Benchmarking Report")
    report.append("")
    report.append(f"**Session ID:** {session_data.get('session_id')}")
    report.append(f"**Start Time:** {session_data.get('start_time', 'N/A')}")
    report.append(f"**End Time:** {session_data.get('end_time', 'N/A')}")
    report.append(f"**Total Configurations:** {len(provider_groups)}")
    report.append(f"**Total Runs:** {len(runs)}")
    report.append("")

    # Summary table
    report.append("## Summary by Provider/Model Configuration")
    report.append("")

    # Build summary table
    table_rows = []
    headers = [
        "Configuration",
        "Runs",
        "Episodes",
        "Avg Duration",
        "Time/Episode",
        "Throughput",
        "Memory/Episode",
        "Success Rate",
    ]

    analyses = []
    for key, group_runs in sorted(provider_groups.items()):
        analysis = analyze_provider_group(group_runs)
        if not analysis or "note" in analysis:
            continue
        analyses.append((key, analysis))

    # Sort by time per episode (fastest first)
    analyses.sort(key=lambda x: x[1].get("seconds_per_episode", float("inf")))

    for key, analysis in analyses:
        provider_info = analysis["provider_info"]
        config_name = generate_provider_key(provider_info)
        run_count = analysis["run_count"]
        total_episodes = analysis["total_episodes"]
        avg_duration = analysis["duration_stats"]["mean"]
        time_per_episode = analysis["seconds_per_episode"]
        throughput = analysis["episodes_per_second"]
        memory_per_ep = analysis.get("memory_per_episode_mb")
        success_rate = analysis["success_rate"]

        table_rows.append(
            [
                config_name[:50],  # Truncate long names
                str(run_count),
                str(total_episodes),
                f"{avg_duration:.1f}s",
                f"{time_per_episode:.1f}s",
                f"{throughput:.3f}/s",
                f"{memory_per_ep:.0f}MB" if memory_per_ep else "N/A",
                f"{success_rate:.1f}%",
            ]
        )

    # Calculate column widths
    col_widths = [
        max(len(h), max((len(row[i]) for row in table_rows), default=0), 3)
        for i, h in enumerate(headers)
    ]

    # Build table
    header_cells = [h.ljust(col_widths[i]) for i, h in enumerate(headers)]
    report.append("| " + " | ".join(header_cells) + " |")
    separator_cells = ["-" * col_widths[i] for i in range(len(headers))]
    report.append("| " + " | ".join(separator_cells) + " |")
    for row in table_rows:
        data_cells = [row[i].ljust(col_widths[i]) for i in range(len(headers))]
        report.append("| " + " | ".join(data_cells) + " |")

    report.append("")

    # Detailed analysis per configuration
    report.append("## Detailed Analysis by Configuration")
    report.append("")

    for key, analysis in analyses:
        provider_info = analysis["provider_info"]
        config_name = generate_provider_key(provider_info)

        report.append(f"### {config_name}")
        report.append("")

        report.append("#### Configuration")
        report.append("")
        report.append(f"- **Speaker Provider:** {provider_info.get('speaker_provider', 'N/A')}")
        report.append(f"- **Speaker Model:** {provider_info.get('speaker_model', 'N/A')}")
        report.append(f"- **Summary Provider:** {provider_info.get('summary_provider', 'N/A')}")
        if provider_info.get("summary_provider") in ("transformers", "local"):
            report.append(
                f"- **Summary Map Model:** {provider_info.get('summary_map_model', 'N/A')}"
            )
            if provider_info.get("summary_reduce_model"):
                report.append(
                    f"- **Summary Reduce Model:** {provider_info.get('summary_reduce_model')}"
                )
        else:
            report.append(f"- **Summary Model:** {provider_info.get('summary_model', 'N/A')}")
        report.append("")

        report.append("#### Performance Metrics")
        report.append("")
        report.append(f"- **Runs:** {analysis['run_count']}")
        report.append(f"- **Total Episodes:** {analysis['total_episodes']}")
        report.append(f"- **Total Duration:** {analysis['total_duration_seconds']:.1f}s")
        report.append(f"- **Average Duration per Run:** {analysis['duration_stats']['mean']:.1f}s")
        report.append(f"- **Time per Episode:** {analysis['seconds_per_episode']:.1f}s")
        report.append(f"- **Throughput:** {analysis['episodes_per_second']:.3f} episodes/second")
        if analysis.get("memory_per_episode_mb"):
            report.append(f"- **Memory per Episode:** {analysis['memory_per_episode_mb']:.0f}MB")
        report.append(f"- **Success Rate:** {analysis['success_rate']:.1f}%")
        report.append("")

        # Duration statistics
        duration_stats = analysis["duration_stats"]
        if duration_stats["count"] > 1:
            report.append("#### Duration Statistics")
            report.append("")
            report.append(f"- **Mean:** {duration_stats['mean']:.1f}s")
            report.append(f"- **Median:** {duration_stats['median']:.1f}s")
            report.append(f"- **Min:** {duration_stats['min']:.1f}s")
            report.append(f"- **Max:** {duration_stats['max']:.1f}s")
            if duration_stats["std_dev"] > 0:
                report.append(f"- **Std Dev:** {duration_stats['std_dev']:.1f}s")
            report.append("")

        # Memory statistics
        if analysis.get("memory_stats"):
            memory_stats = analysis["memory_stats"]
            report.append("#### Memory Statistics")
            report.append("")
            report.append(f"- **Mean Peak Memory:** {memory_stats['mean']:.0f}MB")
            report.append(f"- **Median:** {memory_stats['median']:.0f}MB")
            report.append(f"- **Min:** {memory_stats['min']:.0f}MB")
            report.append(f"- **Max:** {memory_stats['max']:.0f}MB")
            if memory_stats["std_dev"] > 0:
                report.append(f"- **Std Dev:** {memory_stats['std_dev']:.0f}MB")
            report.append("")

    # Comparison section
    if len(analyses) > 1:
        report.append("## Performance Comparison")
        report.append("")

        # Find fastest and slowest
        fastest = min(analyses, key=lambda x: x[1].get("seconds_per_episode", float("inf")))
        slowest = max(analyses, key=lambda x: x[1].get("seconds_per_episode", 0))

        report.append("### Speed Comparison")
        report.append("")
        report.append(f"**Fastest:** {generate_provider_key(fastest[1]['provider_info'])}")
        report.append(f"  - Time per episode: {fastest[1]['seconds_per_episode']:.1f}s")
        report.append("")
        report.append(f"**Slowest:** {generate_provider_key(slowest[1]['provider_info'])}")
        report.append(f"  - Time per episode: {slowest[1]['seconds_per_episode']:.1f}s")
        report.append("")

        if fastest[1].get("seconds_per_episode", 0) > 0:
            speed_ratio = slowest[1]["seconds_per_episode"] / fastest[1]["seconds_per_episode"]
            report.append(f"**Speed Difference:** {speed_ratio:.1f}x slower")
            report.append("")

        # Memory comparison
        memory_analyses = [
            (k, a) for k, a in analyses if a.get("memory_per_episode_mb") is not None
        ]
        if len(memory_analyses) > 1:
            most_memory = max(memory_analyses, key=lambda x: x[1].get("memory_per_episode_mb", 0))
            least_memory = min(
                memory_analyses, key=lambda x: x[1].get("memory_per_episode_mb", float("inf"))
            )

            report.append("### Memory Comparison")
            report.append("")
            report.append(
                f"**Most Memory:** {generate_provider_key(most_memory[1]['provider_info'])}"
            )
            report.append(
                f"  - Memory per episode: {most_memory[1]['memory_per_episode_mb']:.0f}MB"
            )
            report.append("")
            report.append(
                f"**Least Memory:** {generate_provider_key(least_memory[1]['provider_info'])}"
            )
            report.append(
                f"  - Memory per episode: {least_memory[1]['memory_per_episode_mb']:.0f}MB"
            )
            report.append("")

    # Baseline comparison section
    if baseline_comparison:
        report.append("## Baseline Comparison")
        report.append("")
        report.append(f"**Baseline:** {baseline_comparison.get('baseline_id')}")
        regressions = baseline_comparison.get("regressions", 0)
        improvements = baseline_comparison.get("improvements", 0)
        if regressions > 0:
            report.append(f"⚠️ **{regressions} regression(s) detected**")
        if improvements > 0:
            report.append(f"✅ **{improvements} improvement(s) detected**")
        if regressions == 0 and improvements == 0:
            report.append("✅ **No significant changes detected**")
        report.append("")

        # Comparison table
        report.append("### Performance Changes vs Baseline")
        report.append("")

        table_rows = []
        headers = [
            "Configuration",
            "Status",
            "Time/Episode Δ",
            "Throughput Δ",
            "Memory/Episode Δ",
        ]

        for key, comparison in sorted(
            baseline_comparison.get("comparisons", {}).items(),
            key=lambda x: (
                (
                    0
                    if x[1].get("status") == "regression"
                    else (1 if x[1].get("status") == "improvement" else 2)
                ),
                x[1].get("duration_percent_change", 0),
            ),
        ):
            if comparison.get("status") == "no_baseline":
                continue

            provider_info = comparison.get("current", {}).get("provider_info", {})
            config_name = generate_provider_key(provider_info)
            status = comparison.get("status", "ok")
            status_icon = (
                "❌" if status == "regression" else ("✅" if status == "improvement" else "➡️")
            )
            status_text = status_icon + " " + status.upper()

            duration_change = comparison.get("duration_percent_change", 0)
            duration_str = f"{duration_change:+.1f}%"

            throughput_change = comparison.get("throughput_percent_change", 0)
            throughput_str = f"{throughput_change:+.1f}%"

            memory_change = comparison.get("memory_percent_change")
            if memory_change is not None:
                memory_str = f"{memory_change:+.1f}%"
            else:
                memory_str = "N/A"

            table_rows.append(
                [
                    config_name[:40],  # Truncate long names
                    status_text,
                    duration_str,
                    throughput_str,
                    memory_str,
                ]
            )

        if table_rows:
            # Calculate column widths
            col_widths = [
                max(len(h), max((len(row[i]) for row in table_rows), default=0), 3)
                for i, h in enumerate(headers)
            ]

            # Build table
            header_cells = [h.ljust(col_widths[i]) for i, h in enumerate(headers)]
            report.append("| " + " | ".join(header_cells) + " |")
            separator_cells = ["-" * col_widths[i] for i in range(len(headers))]
            report.append("| " + " | ".join(separator_cells) + " |")
            for row in table_rows:
                data_cells = [row[i].ljust(col_widths[i]) for i in range(len(headers))]
                report.append("| " + " | ".join(data_cells) + " |")
            report.append("")

        # Detailed comparison per configuration
        report.append("### Detailed Comparison by Configuration")
        report.append("")

        for key, comparison in sorted(
            baseline_comparison.get("comparisons", {}).items(),
            key=lambda x: (
                (
                    0
                    if x[1].get("status") == "regression"
                    else (1 if x[1].get("status") == "improvement" else 2)
                ),
            ),
        ):
            if comparison.get("status") == "no_baseline":
                continue

            current = comparison.get("current", {})
            baseline = comparison.get("baseline", {})
            provider_info = current.get("provider_info", {})
            config_name = generate_provider_key(provider_info)
            status = comparison.get("status", "ok")

            report.append(f"#### {config_name} - {status.upper()}")
            report.append("")

            # Current vs Baseline metrics
            report.append("**Current vs Baseline:**")
            report.append("")
            report.append(
                f"- **Time per Episode:** {current.get('seconds_per_episode', 0):.1f}s "
                f"(baseline: {baseline.get('seconds_per_episode', 0):.1f}s, "
                f"Δ {comparison.get('duration_percent_change', 0):+.1f}%)"
            )
            report.append(
                f"- **Throughput:** {current.get('episodes_per_second', 0):.3f}/s "
                f"(baseline: {baseline.get('episodes_per_second', 0):.3f}/s, "
                f"Δ {comparison.get('throughput_percent_change', 0):+.1f}%)"
            )
            if comparison.get("memory_percent_change") is not None:
                report.append(
                    f"- **Memory per Episode:** {current.get('memory_per_episode_mb', 0):.0f}MB "
                    f"(baseline: {baseline.get('memory_per_episode_mb', 0):.0f}MB, "
                    f"Δ {comparison.get('memory_percent_change', 0):+.1f}%)"
                )

            if comparison.get("change_reasons"):
                report.append("")
                report.append("**Changes:**")
                for reason in comparison["change_reasons"]:
                    report.append(f"- {reason}")
            report.append("")

    return "\n".join(report)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate performance benchmarking report from acceptance test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Session ID (e.g., 'session_20260206_103000')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".test_outputs/acceptance",
        help="Output directory (default: .test_outputs/acceptance)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="both",
        choices=["markdown", "json", "both"],
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Baseline ID to compare against (optional)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load session data
    output_dir = Path(args.output_dir)
    session_data = load_session_data(args.session_id, output_dir)

    # Baseline comparison
    baseline_comparison = None
    if args.compare_baseline:
        try:
            baseline_data = load_baseline(args.compare_baseline, output_dir)
            runs = session_data.get("runs", [])
            provider_groups = group_runs_by_provider(runs)
            baseline_comparison = compare_provider_groups_with_baseline(
                provider_groups, baseline_data
            )
        except FileNotFoundError as e:
            logger.warning(f"Baseline not found: {e}")

    # Generate report
    markdown_report = generate_benchmark_report(session_data, baseline_comparison)

    # Determine session directory
    normalized_session_id = args.session_id
    if normalized_session_id.startswith("session_"):
        normalized_session_id = normalized_session_id.replace("session_", "", 1)

    session_dir = output_dir / "sessions" / f"session_{normalized_session_id}"
    if not session_dir.exists():
        session_dir = output_dir

    # Save reports
    if args.output_format in ["markdown", "both"]:
        report_path = session_dir / f"benchmark_{normalized_session_id}.md"
        with open(report_path, "w") as f:
            f.write(markdown_report)
        logger.info(f"Benchmark report saved: {report_path}")

    if args.output_format in ["json", "both"]:
        # Generate JSON report
        runs = session_data.get("runs", [])
        provider_groups = group_runs_by_provider(runs)

        json_report = {
            "session_id": args.session_id,
            "configurations": {},
            "baseline_comparison": baseline_comparison,
        }

        for key, group_runs in provider_groups.items():
            analysis = analyze_provider_group(group_runs)
            if analysis and "note" not in analysis:
                config_name = generate_provider_key(analysis["provider_info"])
                json_report["configurations"][config_name] = analysis

        json_path = session_dir / f"benchmark_{normalized_session_id}.json"
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2)
        logger.info(f"JSON benchmark report saved: {json_path}")

    logger.info("Benchmark report generation complete")


if __name__ == "__main__":
    main()
