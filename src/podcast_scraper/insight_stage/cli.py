"""The ``gi`` and ``gi-repair`` subcommands (moved verbatim from ``podcast_scraper.cli``, S7)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, cast, List, Literal, Optional, Sequence

from podcast_scraper import config
from podcast_scraper.cli import _LOGGER
from podcast_scraper.utils.log_redaction import format_exception_for_log


def _run_gi(args: argparse.Namespace, log: Optional[logging.Logger] = None) -> int:
    """Dispatch gi subcommand (validate, inspect, show-insight, explore, query)."""
    logger = log or _LOGGER
    sub = getattr(args, "gi_subcommand", None)
    if sub == "validate":
        return _run_gi_validate(args, logger=logger)
    if sub == "inspect":
        return _run_gi_inspect(args, logger=logger)
    if sub == "show-insight":
        return _run_gi_show_insight(args, logger=logger)
    if sub == "explore":
        return _run_gi_explore(args, logger=logger)
    if sub == "clusters":
        from podcast_scraper.insight_stage.cluster_cli import run_cluster_browse_cli

        return run_cluster_browse_cli(args, logger)
    if sub == "explore-quotes":
        from podcast_scraper.insight_stage.cluster_cli import run_explore_quotes_cli

        return run_explore_quotes_cli(args, logger)
    if sub == "topic-insights":
        from podcast_scraper.insight_stage.cluster_cli import run_topic_insights_cli

        return run_topic_insights_cli(args, logger)
    if sub == "query":
        return _run_gi_query(args, logger=logger)
    if sub == "export":
        return _run_gi_export(args, logger=logger)
    logger.error("Unknown gi subcommand: %s", sub)
    return 1


def _run_gi_export(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Export all gi.json under a run as NDJSON or merged JSON (symmetric with ``kg export``)."""
    import json
    import sys

    from podcast_scraper.gi.contracts import build_gi_corpus_bundle_output
    from podcast_scraper.gi.corpus import export_merged_json, export_ndjson, load_gi_artifacts
    from podcast_scraper.gi.explore import (
        EXIT_INVALID_ARGS,
        EXIT_NO_ARTIFACTS,
        EXIT_SUCCESS,
        scan_artifact_paths,
    )

    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("--output-dir is required")
        return EXIT_INVALID_ARGS
    out_root = Path(output_dir)
    if not out_root.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return EXIT_NO_ARTIFACTS
    paths = scan_artifact_paths(out_root)
    if not paths:
        logger.error("No .gi.json artifacts found under %s", output_dir)
        return EXIT_NO_ARTIFACTS
    strict = getattr(args, "strict", False)
    try:
        loaded = load_gi_artifacts(paths, validate=True, strict=strict)
    except Exception as e:
        logger.error("Validation failed: %s", format_exception_for_log(e))
        return 1
    if not loaded:
        logger.error("No valid artifacts loaded")
        return EXIT_NO_ARTIFACTS
    fmt = getattr(args, "format", "ndjson")
    out_file = getattr(args, "out", None)
    if fmt == "ndjson":
        if out_file:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:

                def _write_ndj(s: str) -> None:
                    f.write(s)

                export_ndjson(loaded, output_dir=out_root, stream_write=_write_ndj)
            logger.info("Wrote NDJSON to %s", out_file)
        else:

            def _write_stdout(s: str) -> None:
                sys.stdout.write(s)

            export_ndjson(loaded, output_dir=out_root, stream_write=_write_stdout)
        return EXIT_SUCCESS
    bundle = export_merged_json(loaded, output_dir=out_root)
    validated = build_gi_corpus_bundle_output(bundle)
    text = json.dumps(validated.model_dump(mode="json"), indent=2, ensure_ascii=False)
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(text, encoding="utf-8")
        logger.info("Wrote merged JSON to %s", out_file)
    else:
        print(text)
    return EXIT_SUCCESS


def _run_gi_validate(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Validate .gi.json files (parity with ``kg validate``)."""
    from podcast_scraper.gi.explore import EXIT_INVALID_ARGS, EXIT_NO_ARTIFACTS, EXIT_SUCCESS
    from podcast_scraper.gi.io import collect_gi_paths_from_inputs, read_artifact

    paths_arg = list(getattr(args, "paths", None) or [])
    if not paths_arg:
        logger.error("Provide one or more paths to .gi.json files or directories")
        return EXIT_INVALID_ARGS
    try:
        paths = collect_gi_paths_from_inputs([Path(p) for p in paths_arg])
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", format_exception_for_log(e))
        return EXIT_INVALID_ARGS
    if not paths:
        logger.error("No .gi.json files found")
        return EXIT_NO_ARTIFACTS
    strict = getattr(args, "strict", False)
    quiet = getattr(args, "quiet", False)
    failed = 0
    for path in paths:
        try:
            read_artifact(path, validate=True, strict=strict)
            if not quiet:
                print(f"OK {path}")
        except Exception as e:
            failed += 1
            logger.error("FAIL %s: %s", path, format_exception_for_log(e))
    if failed:
        logger.error("%s of %s file(s) failed validation", failed, len(paths))
        return 1
    if not quiet:
        print(f"All {len(paths)} file(s) passed validation.")
    return EXIT_SUCCESS


def _resolve_artifact_path_gi(args: argparse.Namespace) -> Optional[Path]:
    """Resolve path to .gi.json from --episode-path or --output-dir + --episode-id."""
    ep_path = getattr(args, "episode_path", None)
    output_dir = getattr(args, "output_dir", None)
    episode_id = getattr(args, "episode_id", None)

    if ep_path:
        p = Path(ep_path)
        if p.is_file() and p.suffix == ".json" and ".gi.json" in p.name:
            return p
        if p.is_dir():
            for f in p.glob("*.gi.json"):
                return f
        if p.is_file():
            return p
        return None

    if output_dir and episode_id:
        from podcast_scraper.gi import find_artifact_by_episode_id

        fid = getattr(args, "feed_id", None)
        return find_artifact_by_episode_id(Path(output_dir), episode_id, feed_id=fid)
    return None


def _run_gi_inspect(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run gi inspect: load artifact, build InspectOutput, print."""
    from podcast_scraper.gi import build_inspect_output, load_artifact_and_transcript

    artifact_path = _resolve_artifact_path_gi(args)
    if not artifact_path:
        episode_id = getattr(args, "episode_id", None)
        episode_path = getattr(args, "episode_path", None)
        output_dir = getattr(args, "output_dir", None)
        if not episode_id and not episode_path:
            logger.error("Provide --episode-id with --output-dir, or --episode-path")
        elif output_dir and not episode_id:
            logger.error("Provide --episode-id when using --output-dir")
        elif output_dir and episode_id:
            from podcast_scraper.utils.corpus_episode_paths import list_artifact_paths_for_episode

            matches = list_artifact_paths_for_episode(
                Path(output_dir),
                episode_id,
                feed_id=None,
                kind="gi",
            )
            if len(matches) > 1 and not getattr(args, "feed_id", None):
                logger.error(
                    "Multiple GI artifacts for episode_id=%s (%d matches); use --feed-id "
                    "(metadata feed.feed_id) to choose one.",
                    episode_id,
                    len(matches),
                )
            else:
                logger.error("Artifact not found for the given episode")
        else:
            logger.error("Artifact not found for the given episode")
        return 1

    try:
        artifact, transcript_text, _ = load_artifact_and_transcript(
            artifact_path,
            validate=True,
            strict=getattr(args, "strict", False),
            load_transcript=True,
        )
    except FileNotFoundError as e:
        logger.error("%s", format_exception_for_log(e))
        return 1
    except ValueError as e:
        logger.error("Validation failed: %s", format_exception_for_log(e))
        return 1

    out = build_inspect_output(artifact, transcript_text)
    fmt = getattr(args, "format", "pretty")
    show = getattr(args, "show", False)
    stats = getattr(args, "stats", True)

    if fmt == "json":
        print(out.model_dump_json(indent=2))
        return 0

    # Pretty
    lines = [
        f"Episode: {out.episode_id}",
        f"Schema: {out.schema_version}  Model: {out.model_version}",
    ]
    if stats and out.stats:
        s = out.stats
        lines.append(
            f"Insights: {s.get('insight_count', 0)}  "
            f"Grounded: {s.get('grounded_count', 0)}  "
            f"Ungrounded: {s.get('ungrounded_count', 0)}  "
            f"Quotes: {s.get('quote_count', 0)}"
        )
    for ins in out.insights:
        lines.append(f"  [{ins.insight_id}] grounded={ins.grounded}")
        if show or ins.supporting_quotes:
            lines.append(f"    {ins.text[:200]}{'...' if len(ins.text) > 200 else ''}")
            for q in ins.supporting_quotes:
                lines.append(
                    f"    → {q.quote_id}: {q.text[:100]}{'...' if len(q.text) > 100 else ''}"
                )
    print("\n".join(lines))
    return 0


def _run_gi_show_insight(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run gi show-insight: find insight by id, print with quotes and evidence."""
    from podcast_scraper.gi import (
        build_inspect_output,
        find_artifact_by_insight_id,
        load_artifact_and_transcript,
    )

    insight_id = getattr(args, "id", None)
    if not insight_id:
        logger.error("--id INSIGHT_ID is required")
        return 1

    artifact_path = _resolve_artifact_path_gi(args)
    if not artifact_path:
        output_dir = getattr(args, "output_dir", None)
        if output_dir:
            artifact_path = find_artifact_by_insight_id(Path(output_dir), insight_id)
        if not artifact_path:
            logger.error(
                "Provide --episode-path to the .gi.json file, or --output-dir to scan for "
                "artifact containing insight %s",
                insight_id,
            )
            return 1

    try:
        artifact, transcript_text, _ = load_artifact_and_transcript(
            artifact_path,
            validate=True,
            strict=False,
            load_transcript=True,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", format_exception_for_log(e))
        return 1

    out = build_inspect_output(artifact, transcript_text)
    insight = next((i for i in out.insights if i.insight_id == insight_id), None)
    if not insight:
        logger.error("Insight %s not found in artifact", insight_id)
        return 1

    fmt = getattr(args, "format", "pretty")
    context_chars = getattr(args, "context_chars", 80)

    if fmt == "json":
        print(insight.model_dump_json(indent=2))
        return 0

    lines = [
        f"Insight: {insight.insight_id}",
        f"  grounded={insight.grounded}  episode_id={insight.episode_id}",
        f"  {insight.text}",
        "Supporting quotes:",
    ]
    for q in insight.supporting_quotes:
        lines.append(f"  [{q.quote_id}] {q.text}")
        if q.evidence.excerpt:
            ev = q.evidence
            lines.append("    evidence: " + ev.transcript_ref)
            lines.append(f"      span: [{ev.char_start}:{ev.char_end}]")
            lines.append(f"    excerpt: {q.evidence.excerpt}")
            if transcript_text and context_chars > 0:
                start = max(0, q.evidence.char_start - context_chars)
                end = min(len(transcript_text), q.evidence.char_end + context_chars)
                lines.append(f"    context: ...{transcript_text[start:end]}...")
    print("\n".join(lines))
    return 0


def _sort_by_evidence_density(insights: list, out_path: Path) -> list:
    """Re-sort insights by evidence density (cluster_size × quote_count)."""
    from podcast_scraper.insight_stage.insight_cluster_context import load_insight_clusters

    ic_index = load_insight_clusters(out_path / "search" / "insight_clusters.json")

    def _density(ins: Any) -> float:
        ic = ic_index.get(ins.insight_id, {})
        cluster_size = ic.get("member_count", 1)
        return float(cluster_size * max(len(ins.supporting_quotes), 1))

    insights.sort(key=_density, reverse=True)
    return insights


def _run_gi_explore(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run gi explore: scan output_dir, filter by topic/speaker, print insights with quotes."""
    import json

    from podcast_scraper.gi.explore import (
        aggregate_topic_entries_for_insights,
        build_explore_output,
        EXIT_INVALID_ARGS,
        EXIT_NO_ARTIFACTS,
        EXIT_NO_RESULTS,
        EXIT_STRICT_VALIDATION_FAILED,
        EXIT_SUCCESS,
        explore_output_to_rfc_dict,
        explore_resolve_insights_and_loaded,
        ExploreValidationError,
        scan_artifact_paths,
    )

    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("--output-dir is required for gi explore")
        return EXIT_INVALID_ARGS
    out_path = Path(output_dir)
    if not out_path.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return EXIT_NO_ARTIFACTS

    paths = scan_artifact_paths(out_path)
    if not paths:
        logger.error("No .gi.json artifacts found under %s", output_dir)
        return EXIT_NO_ARTIFACTS

    strict = getattr(args, "strict", False)
    topic = getattr(args, "topic", None)
    speaker = getattr(args, "speaker", None)
    grounded_only = getattr(args, "grounded_only", False)
    min_confidence = getattr(args, "min_confidence", None)
    limit = getattr(args, "limit", 50)
    sort_by = getattr(args, "explore_sort", "confidence")

    # evidence-density sort uses confidence as base, then re-sorts
    resolve_sort = "confidence" if sort_by == "evidence-density" else sort_by

    try:
        insights, _semantic_ranked, loaded, ep_searched = explore_resolve_insights_and_loaded(
            out_path,
            paths,
            topic=topic,
            speaker=speaker,
            grounded_only=grounded_only,
            min_confidence=min_confidence,
            sort_by=cast(Literal["confidence", "time"], resolve_sort),
            strict=strict,
        )
    except ExploreValidationError as e:
        logger.error(
            "Strict validation failed for %s: %s",
            e.path,
            format_exception_for_log(e),
        )
        return EXIT_STRICT_VALIDATION_FAILED

    if sort_by == "evidence-density":
        insights = _sort_by_evidence_density(insights, out_path)

    if limit and limit > 0:
        insights = insights[:limit]

    # Build cluster lookup if requested
    expand_clusters = getattr(args, "expand_clusters", False)
    cluster_lookup: dict = {}
    if expand_clusters:
        from podcast_scraper.insight_stage.insight_cluster_context import (
            expand_with_cluster_context,
            load_insight_clusters,
        )

        clusters_path = out_path / "search" / "insight_clusters.json"
        cluster_index = load_insight_clusters(clusters_path)
        if cluster_index:
            # Convert InsightSummary objects to dicts for cluster expansion
            ins_dicts = [{"insight_id": i.insight_id, "episode_id": i.episode_id} for i in insights]
            expanded = expand_with_cluster_context(ins_dicts, cluster_index=cluster_index)
            for ins_dict in expanded:
                if "cluster" in ins_dict:
                    cluster_lookup[ins_dict["insight_id"]] = ins_dict["cluster"]

    topic_rows = aggregate_topic_entries_for_insights(loaded, insights)
    explore_out = build_explore_output(
        insights,
        episodes_searched=ep_searched,
        topic=topic,
        speaker_filter=speaker,
        topics=topic_rows,
    )

    out_format = getattr(args, "format", "pretty")
    out_file = getattr(args, "out", None)
    if out_format == "json":
        rfc_dict = explore_output_to_rfc_dict(explore_out)
        # Inject cluster context into JSON output
        if expand_clusters and cluster_lookup:
            for ins_entry in rfc_dict.get("insights", []):
                iid = ins_entry.get("insight_id", "")
                if iid in cluster_lookup:
                    ins_entry["cluster"] = cluster_lookup[iid]
        payload = json.dumps(rfc_dict, indent=2)
    else:
        lines = [
            f"Topic: {explore_out.topic or '(all)'}",
            f"Speaker filter: {explore_out.speaker_filter or '(none)'}",
            f"Episodes searched: {explore_out.episodes_searched}",
            f"Insights: {explore_out.summary.get('insight_count', 0)} "
            f"(grounded: {explore_out.summary.get('grounded_insight_count', 0)})",
            f"Quotes: {explore_out.summary.get('quote_count', 0)}",
            f"Distinct speakers (quotes): {explore_out.summary.get('speaker_count', 0)}",
            "",
        ]
        if explore_out.top_speakers:
            lines.append("Top speakers (by quote count):")
            for ts in explore_out.top_speakers[:10]:
                sid = str(ts.speaker_id)
                nm = (ts.name or "").strip()
                if nm and nm.casefold() != sid.casefold():
                    label = f"{sid} ({nm})"
                else:
                    label = sid
                lines.append(f"  {label}: quotes={ts.quote_count}, insights={ts.insight_count}")
            lines.append("")
        for ins in explore_out.insights:
            ep_title = ins.episode_title or ""
            lines.append(
                f"[{ins.insight_id}] {ins.episode_id}"
                f"{(' — ' + ep_title) if ep_title else ''} grounded={ins.grounded}"
            )
            lines.append(f"  {ins.text}")
            for q in ins.supporting_quotes:
                lines.append(
                    f"    quote: {q.text[:80]}..." if len(q.text) > 80 else f"    quote: {q.text}"
                )
            # Show cluster context if expanded
            if expand_clusters and ins.insight_id in cluster_lookup:
                from podcast_scraper.insight_stage.insight_cluster_context import (
                    format_cluster_context,
                )

                ctx = format_cluster_context({"cluster": cluster_lookup[ins.insight_id]})
                if ctx:
                    lines.append(ctx)
        payload = "\n".join(lines)

    if out_file:
        Path(out_file).write_text(payload, encoding="utf-8")
        logger.info("Wrote output to %s", out_file)
    else:
        print(payload)

    if not insights and (topic or speaker):
        return EXIT_NO_RESULTS
    return EXIT_SUCCESS


def _run_gi_query(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run gi query: UC4 natural-language patterns → explore-style RFC JSON answer."""
    import json

    from podcast_scraper.gi.explore import (
        EXIT_INVALID_ARGS,
        EXIT_NO_ARTIFACTS,
        EXIT_SUCCESS,
        run_uc4_semantic_qa,
        scan_artifact_paths,
    )

    output_dir = getattr(args, "output_dir", None)
    question = getattr(args, "question", None)
    if not output_dir or not question or not str(question).strip():
        logger.error("--output-dir and --question are required for gi query")
        return EXIT_INVALID_ARGS
    out_path = Path(output_dir)
    if not out_path.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return EXIT_NO_ARTIFACTS
    if not scan_artifact_paths(out_path):
        logger.error("No .gi.json artifacts found under %s", output_dir)
        return EXIT_NO_ARTIFACTS

    limit = getattr(args, "query_limit", 20)
    strict = getattr(args, "strict", False)
    result = run_uc4_semantic_qa(out_path, question.strip(), limit=limit, strict=strict)
    if result is None:
        logger.error(
            "Question did not match a supported pattern. Examples: "
            "'What insights about X?', 'What insights are there about X?', "
            "'What did Y say?', 'What did Y say about X?', "
            "'Which topics have the most insights?', 'Top topics'."
        )
        return EXIT_INVALID_ARGS

    fmt = getattr(args, "format", "json")
    if fmt == "json":
        print(json.dumps(result, indent=2))
    else:
        print(result.get("explanation", ""))
        answer_payload = result.get("answer")
        if isinstance(answer_payload, dict):
            summ = answer_payload.get("summary") or {}
            print(
                f"Insights: {summ.get('insight_count', 0)}  "
                f"Episodes searched: {answer_payload.get('episodes_searched', 0)}"
            )
    return EXIT_SUCCESS


def _parse_gi_args(gi_argv: Sequence[str]) -> argparse.Namespace:
    """Parse 'gi' subcommand arguments (validate, inspect, show-insight, explore, query)."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper gi",
        description="Inspect Grounded Insight Layer (GIL) artifacts (gi.json).",
    )
    subparsers = parser.add_subparsers(dest="gi_subcommand", required=True, help="Command")

    val = subparsers.add_parser(
        "validate",
        help="Validate .gi.json files against schema (use --strict for full JSON Schema)",
    )
    val.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="Files or directories containing .gi.json",
    )
    val.add_argument(
        "--strict",
        action="store_true",
        help="Full JSON Schema validation (docs/architecture/gi/gi.schema.json)",
    )
    val.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print failures",
    )

    # gi inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect one episode's GIL artifact")
    inspect_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory containing metadata/*.gi.json",
    )
    inspect_parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Episode ID (artifact episode_id field); required if not using --episode-path",
    )
    inspect_parser.add_argument(
        "--feed-id",
        type=str,
        default=None,
        help="Feed scope (metadata feed.feed_id) when multiple feeds share an episode id",
    )
    inspect_parser.add_argument(
        "--episode-path",
        type=str,
        default=None,
        help="Path to .gi.json file (or directory containing one .gi.json)",
    )
    inspect_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if artifact does not pass schema validation",
    )
    inspect_parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    inspect_parser.add_argument(
        "--show",
        action="store_true",
        help="Show full insight text and supporting quotes",
    )
    inspect_parser.add_argument(
        "--stats",
        action="store_true",
        default=True,
        help="Show summary stats (default: True)",
    )
    inspect_parser.add_argument("--no-stats", dest="stats", action="store_false")

    # gi show-insight
    show_parser = subparsers.add_parser(
        "show-insight",
        help="Show one insight by ID with quotes and evidence",
    )
    show_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="INSIGHT_ID",
        help="Insight node id from gi.json (e.g. insight:a1b2c3d4e5f67890)",
    )
    show_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory containing metadata/*.gi.json",
    )
    show_parser.add_argument(
        "--episode-path",
        type=str,
        default=None,
        help="Path to .gi.json file for the episode",
    )
    show_parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    show_parser.add_argument(
        "--context-chars",
        type=int,
        default=80,
        metavar="N",
        help="Characters of context around evidence span (default: 80)",
    )

    # gi explore
    explore_parser = subparsers.add_parser(
        "explore",
        help="Cross-episode query: insights about a topic with supporting quotes",
    )
    explore_parser.add_argument(
        "--topic",
        type=str,
        default=None,
        metavar="LABEL",
        help="Topic filter (match Topic label or substring in insight text)",
    )
    explore_parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        metavar="SUBSTRING",
        help="Speaker filter: substring match on quote speaker_id or graph speaker name (UC2)",
    )
    explore_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Output directory containing metadata/*.gi.json",
    )
    explore_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Max number of insights to return (default: 50)",
    )
    explore_parser.add_argument(
        "--grounded-only",
        action="store_true",
        help="Only return grounded insights",
    )
    explore_parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        metavar="0..1",
        help=(
            "Minimum insight confidence (0.0-1.0). Above 0, insights with no confidence are "
            "left out; 0 keeps them"
        ),
    )
    explore_parser.add_argument(
        "--sort",
        choices=("confidence", "time", "evidence-density"),
        default="confidence",
        dest="explore_sort",
        help=(
            "Sort insights by confidence (desc), episode publish_date (desc), "
            "or evidence-density (cluster size × quote count, desc)"
        ),
    )
    explore_parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    explore_parser.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="PATH",
        help="Write output to file (optional)",
    )
    explore_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first artifact that fails schema validation (exit 5)",
    )
    explore_parser.add_argument(
        "--expand-clusters",
        action="store_true",
        help="Expand each insight with cross-episode cluster context (insight_clusters.json)",
    )

    # gi clusters (#601 3b: browse insight clusters)
    clusters_parser = subparsers.add_parser(
        "clusters",
        help="Browse insight clusters sorted by member count",
    )
    clusters_parser.add_argument(
        "--output-dir", type=str, required=True, help="Pipeline output directory"
    )
    clusters_parser.add_argument(
        "--top", type=int, default=10, metavar="N", help="Top N clusters (default: 10)"
    )
    clusters_parser.add_argument(
        "--format", choices=("pretty", "json"), default="pretty", help="Output format"
    )

    # gi explore-quotes (#601 3c: quote-level semantic search)
    eq_parser = subparsers.add_parser(
        "explore-quotes",
        help="Search quotes directly via semantic (LanceDB) search",
    )
    eq_parser.add_argument("--query", type=str, required=True, help="Search query")
    eq_parser.add_argument(
        "--output-dir", type=str, required=True, help="Pipeline output directory"
    )
    eq_parser.add_argument(
        "--top-k", type=int, default=10, metavar="N", help="Number of results (default: 10)"
    )
    eq_parser.add_argument(
        "--format", choices=("pretty", "json"), default="pretty", help="Output format"
    )

    # gi topic-insights (#601 3d: topic × insight matrix)
    ti_parser = subparsers.add_parser(
        "topic-insights",
        help="Show insight clusters associated with a topic via ABOUT edges",
    )
    ti_parser.add_argument("--topic", type=str, required=True, help="Topic label or substring")
    ti_parser.add_argument(
        "--output-dir", type=str, required=True, help="Pipeline output directory"
    )
    ti_parser.add_argument(
        "--format", choices=("pretty", "json"), default="pretty", help="Output format"
    )

    # gi query (UC4: tiny pattern map → explore RFC answer)
    query_parser = subparsers.add_parser(
        "query",
        help="Natural-language question → matched explore result",
    )
    query_parser.add_argument(
        "--question",
        type=str,
        required=True,
        metavar="TEXT",
        help="Question (e.g. 'What insights about inflation?' or 'What did Sam say?')",
    )
    query_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Output directory containing metadata/*.gi.json",
    )
    query_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        metavar="N",
        dest="query_limit",
        help="Max insights in the answer (default: 20)",
    )
    query_parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="json",
        help="Output format (default: json)",
    )
    query_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first artifact that fails schema validation (exit 5)",
    )

    gi_exp = subparsers.add_parser(
        "export",
        help="Export all gi.json under a run as NDJSON or merged JSON bundle",
    )
    gi_exp.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Pipeline output directory to scan for .gi.json",
    )
    gi_exp.add_argument(
        "--format",
        choices=("ndjson", "merged"),
        default="ndjson",
        help="ndjson: one artifact per line; merged: single JSON document (default: ndjson)",
    )
    gi_exp.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="PATH",
        help="Write to file (default: stdout)",
    )
    gi_exp.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any artifact fails strict schema validation",
    )

    args = parser.parse_args(gi_argv)
    args.command = "gi"
    return args


def _parse_gi_repair_argv(argv: List[str]) -> argparse.Namespace:
    """``gi-repair`` — re-derive placeholder GI artifacts in place (#1655)."""
    parser = argparse.ArgumentParser(
        prog="podcast-scraper gi-repair",
        description=(
            "Re-derive GI for episodes carrying a pre-#1657 placeholder insight, rewriting the "
            "SAME gi.json path. Corpus-driven and standalone: no RSS fetch, no new run dir. "
            "A per-episode failure leaves that placeholder intact and exits non-zero."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Corpus parent path")
    parser.add_argument("--config", default=None, help="Profile/config providing the GI providers")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be repaired and write nothing.",
    )
    parser.add_argument(
        "--audit-file",
        default=None,
        help="JSONL audit trail (default: <output-dir>/gi_repair_report.jsonl)",
    )
    parser.add_argument(
        "--litellm-api-base",
        type=str,
        default=None,
        help=(
            "LiteLLM gateway base URL, overriding the profile's value. gi-repair builds its "
            "config from --config alone, and cloud_balanced pins the HOMELAB gateway — so "
            "without this a prod repair bills homelab instead of prod's own gateway "
            "(ADR-142, not in this repository)"
        ),
    )
    parser.add_argument(
        "--episode-ids",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "File of episode_ids (one per line) to re-derive, instead of the "
            "legacy-placeholder work-list. Selection by IDENTITY rather than by damage: "
            "re-deriving the SAME episode under two configs is what makes a prompt / model / "
            "rater change measurable. Requires --force-healthy for artifacts that are fine"
        ),
    )
    parser.add_argument(
        "--force-healthy",
        action="store_true",
        help=(
            "Re-derive artifacts that are NOT legacy placeholders, overwriting good work in "
            "place. Off by default because that refusal is the safety property of a "
            "corpus-wide sweep; needed for deliberate before/after comparison on a known-good "
            "episode. Logs a WARNING per artifact so the choice is visible in the audit trail"
        ),
    )
    args = parser.parse_args(argv)
    args.command = "gi-repair"
    return args


def _run_gi_repair_cli(args: argparse.Namespace, log: logging.Logger) -> int:
    """Run the repair and return the process exit code (0 only when every episode succeeded)."""
    from pathlib import Path as _Path

    from podcast_scraper.gi.repair import repair_placeholder_artifacts

    corpus_root = _Path(args.output_dir).expanduser()
    if not corpus_root.is_dir():
        print(f"Error: --output-dir is not a directory: {corpus_root}", file=sys.stderr)
        return 2

    cfg = None
    if getattr(args, "config", None):
        raw = config.load_config_file(args.config)
        data = dict(raw) if isinstance(raw, dict) else dict(getattr(raw, "__dict__", {}))
        data.setdefault("rss_url", "https://example.invalid/feed.xml")
        # Same override the main CLI grew in #1676, for the same reason: this entry point
        # reads the profile directly, so the profile's homelab pin would otherwise apply to a
        # production repair. Guarded on `is not None` so an absent flag never clobbers it.
        if getattr(args, "litellm_api_base", None) is not None:
            data["litellm_api_base"] = args.litellm_api_base
        # gi-repair re-derives from transcripts already on disk and calls no ASR provider. Say
        # so, rather than inheriting the profile's transcription posture: cloud_balanced pins
        # transcription_provider=deepgram, whose validator then demanded a Deepgram key for a
        # stage this command never runs. Declaring the truth is better than either carrying a
        # credential we do not use or loosening the validator for everyone.
        data["transcribe_missing"] = False
        cfg = config.Config(**data)

    audit = _Path(args.audit_file) if args.audit_file else corpus_root / "gi_repair_report.jsonl"

    episode_ids: Optional[List[str]] = None
    ids_file = getattr(args, "episode_ids", None)
    if ids_file:
        try:
            raw_lines = Path(ids_file).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.error("cannot read --episode-ids %s: %s", ids_file, exc)
            return 2
        # ``splitlines`` already handles CRLF; ``strip`` also drops a stray BOM-adjacent space.
        raw_ids = [ln.strip() for ln in raw_lines]
        raw_ids = [i for i in raw_ids if i and not i.startswith("#")]
        # Order-preserving dedupe. Every id is a paid LLM re-derivation, so a duplicated line —
        # trivially produced by appending to a work-list twice — is money and DGX time spent to
        # overwrite an artifact with a second sample of itself. Report the drop rather than
        # silently collapsing, so a work-list that was not what the operator thought is visible.
        episode_ids = list(dict.fromkeys(raw_ids))
        if len(episode_ids) != len(raw_ids):
            log.warning(
                "--episode-ids %s: dropped %d duplicate id(s) — %d unique of %d listed",
                ids_file,
                len(raw_ids) - len(episode_ids),
                len(episode_ids),
                len(raw_ids),
            )
        if not episode_ids:
            log.error("--episode-ids %s contained no ids", ids_file)
            return 2
        log.info("gi-repair: targeting %d episode id(s) from %s", len(episode_ids), ids_file)

    report = repair_placeholder_artifacts(
        corpus_root,
        cfg,
        dry_run=bool(args.dry_run),
        audit_path=None if args.dry_run else audit,
        episode_ids=episode_ids,
        force_healthy=bool(getattr(args, "force_healthy", False)),
    )
    print(report.format())
    if not args.dry_run:
        print(f"\naudit trail: {audit}")
    log.info("gi-repair finished: %d repaired, %d failed", len(report.repaired), len(report.failed))
    return 0 if report.ok else 1
