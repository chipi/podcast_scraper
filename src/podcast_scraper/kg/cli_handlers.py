"""CLI handlers for `kg` subcommands (validate, inspect, export, entities, topics)."""

from __future__ import annotations

import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

from podcast_scraper.utils.log_redaction import format_exception_for_log

from .contracts import (
    build_kg_corpus_bundle_output,
    build_kg_entity_rollup_output,
    build_kg_inspect_output,
    build_kg_topic_pairs_output,
)
from .corpus import (
    collect_kg_paths_from_inputs,
    entity_rollup,
    EXIT_INVALID_ARGS,
    EXIT_NO_ARTIFACTS,
    EXIT_SUCCESS,
    export_merged_json,
    export_ndjson,
    inspect_summary,
    load_kg_artifacts,
    scan_kg_artifact_paths,
    topic_cooccurrence,
)
from .io import read_artifact
from .load import find_kg_artifact_by_episode_id
from .schema import validate_artifact


def resolve_kg_artifact_path(args: Namespace) -> Optional[Path]:
    """Resolve path to .kg.json from --episode-path or --output-dir + --episode-id."""
    ep_path = getattr(args, "episode_path", None)
    output_dir = getattr(args, "output_dir", None)
    episode_id = getattr(args, "episode_id", None)

    if ep_path:
        p = Path(ep_path)
        if p.is_file() and p.name.endswith(".kg.json"):
            return p
        if p.is_dir():
            for f in p.glob("*.kg.json"):
                return f
        if p.is_file():
            return p
        return None

    if output_dir and episode_id:
        return find_kg_artifact_by_episode_id(Path(output_dir), episode_id)
    return None


def run_kg_validate(args: Namespace, logger: logging.Logger) -> int:
    """Validate .kg.json files (strict JSON Schema when --strict)."""
    paths_arg: List[str] = list(getattr(args, "paths", None) or [])
    if not paths_arg:
        logger.error("Provide one or more paths to .kg.json files or directories")
        return EXIT_INVALID_ARGS
    try:
        paths = collect_kg_paths_from_inputs([Path(p) for p in paths_arg])
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", format_exception_for_log(e))
        return EXIT_INVALID_ARGS
    if not paths:
        logger.error("No .kg.json files found")
        return EXIT_NO_ARTIFACTS
    strict = getattr(args, "strict", False)
    quiet = getattr(args, "quiet", False)
    failed = 0
    for path in paths:
        try:
            data = read_artifact(path)
            validate_artifact(data, strict=strict)
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


def run_kg_inspect(args: Namespace, logger: logging.Logger) -> int:
    """Print summary for one KG artifact."""
    artifact_path = resolve_kg_artifact_path(args)
    if not artifact_path:
        episode_id = getattr(args, "episode_id", None)
        episode_path = getattr(args, "episode_path", None)
        output_dir = getattr(args, "output_dir", None)
        if not episode_id and not episode_path:
            logger.error("Provide --episode-id with --output-dir, or --episode-path")
        elif output_dir and not episode_id:
            logger.error("Provide --episode-id when using --output-dir")
        else:
            logger.error("KG artifact not found for the given episode")
        return 1
    try:
        data = read_artifact(artifact_path)
        validate_artifact(data, strict=getattr(args, "strict", False))
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", format_exception_for_log(e))
        return 1
    summary = inspect_summary(data, artifact_path=artifact_path)
    fmt = getattr(args, "format", "pretty")
    if fmt == "json":
        out = build_kg_inspect_output(data, artifact_path=artifact_path)
        print(json.dumps(out.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return EXIT_SUCCESS
    ext = summary.get("extraction") or {}
    lines = [
        f"Episode: {summary.get('episode_id')}",
        f"Schema: {summary.get('schema_version')}",
        f"Artifact: {summary.get('artifact_path')}",
        (
            f"Extraction: {ext.get('model_version', '')} @ {ext.get('extracted_at', '')} "
            f"(transcript: {ext.get('transcript_ref', '')})"
        ),
        (
            f"Nodes: {summary.get('node_count', 0)}  "
            f"Edges: {summary.get('edge_count', 0)}  "
            f"By type: {summary.get('nodes_by_type', {})}"
        ),
    ]
    if summary.get("episode_title"):
        lines.append(f"Title: {summary['episode_title']}")
    for t in summary.get("topics") or []:
        lines.append(f"  Topic: {t.get('label')} ({t.get('slug')}) [{t.get('id')}]")
    for ent in summary.get("entities") or []:
        role = ent.get("role")
        r = f" role={role}" if role else ""
        lines.append(f"  Entity: {ent.get('name')} ({ent.get('entity_kind')}){r} [{ent.get('id')}]")
    print("\n".join(lines))
    return EXIT_SUCCESS


def run_kg_export(args: Namespace, logger: logging.Logger) -> int:
    """Export corpus as NDJSON or merged JSON."""
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("--output-dir is required")
        return EXIT_INVALID_ARGS
    out_root = Path(output_dir)
    if not out_root.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return EXIT_NO_ARTIFACTS
    paths = scan_kg_artifact_paths(out_root)
    if not paths:
        logger.error("No .kg.json artifacts found under %s", output_dir)
        return EXIT_NO_ARTIFACTS
    strict = getattr(args, "strict", False)
    try:
        loaded = load_kg_artifacts(paths, validate=True, strict=strict)
    except ValueError as e:
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
    validated = build_kg_corpus_bundle_output(bundle)
    text = json.dumps(validated.model_dump(mode="json"), indent=2, ensure_ascii=False)
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(text, encoding="utf-8")
        logger.info("Wrote merged JSON to %s", out_file)
    else:
        print(text)
    return EXIT_SUCCESS


def run_kg_entities(args: Namespace, logger: logging.Logger) -> int:
    """Entity roll-up across a corpus directory."""
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("--output-dir is required")
        return EXIT_INVALID_ARGS
    out_root = Path(output_dir)
    if not out_root.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return EXIT_NO_ARTIFACTS
    paths = scan_kg_artifact_paths(out_root)
    if not paths:
        logger.error("No .kg.json artifacts found under %s", output_dir)
        return EXIT_NO_ARTIFACTS
    strict = getattr(args, "strict", False)
    try:
        loaded = load_kg_artifacts(paths, validate=True, strict=strict)
    except ValueError as e:
        logger.error("Validation failed: %s", format_exception_for_log(e))
        return 1
    min_ep = int(getattr(args, "min_episodes", 1) or 1)
    rows = entity_rollup(loaded, min_episodes=min_ep, output_dir=out_root)
    fmt = getattr(args, "format", "pretty")
    if fmt == "json":
        out = build_kg_entity_rollup_output(rows)
        print(json.dumps(out.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return EXIT_SUCCESS
    lines = [f"Entities (min_episodes={min_ep}, n={len(rows)}):", ""]
    for r in rows:
        lines.append(
            f"{r['name']} ({r['entity_kind']}) — {r['episode_count']} episodes, "
            f"{r['mention_count']} mentions"
        )
    print("\n".join(lines))
    return EXIT_SUCCESS


def run_kg_topics(args: Namespace, logger: logging.Logger) -> int:
    """Topic pair co-occurrence across episodes."""
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("--output-dir is required")
        return EXIT_INVALID_ARGS
    out_root = Path(output_dir)
    if not out_root.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return EXIT_NO_ARTIFACTS
    paths = scan_kg_artifact_paths(out_root)
    if not paths:
        logger.error("No .kg.json artifacts found under %s", output_dir)
        return EXIT_NO_ARTIFACTS
    strict = getattr(args, "strict", False)
    try:
        loaded = load_kg_artifacts(paths, validate=True, strict=strict)
    except ValueError as e:
        logger.error("Validation failed: %s", format_exception_for_log(e))
        return 1
    min_sup = int(getattr(args, "min_support", 1) or 1)
    rows = topic_cooccurrence(loaded, min_support=min_sup)
    fmt = getattr(args, "format", "pretty")
    if fmt == "json":
        out = build_kg_topic_pairs_output(rows)
        print(json.dumps(out.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return EXIT_SUCCESS
    lines = [f"Topic pairs (min_support={min_sup}, n={len(rows)}):", ""]
    for r in rows:
        lines.append(
            f"{r['topic_a_label']} + {r['topic_b_label']} — {r['episode_count']} episode(s)"
        )
    print("\n".join(lines))
    return EXIT_SUCCESS


def run_kg(args: Namespace, logger: logging.Logger) -> int:
    """Dispatch kg subcommand."""
    sub = getattr(args, "kg_subcommand", None)
    if sub == "validate":
        return run_kg_validate(args, logger)
    if sub == "inspect":
        return run_kg_inspect(args, logger)
    if sub == "export":
        return run_kg_export(args, logger)
    if sub == "entities":
        return run_kg_entities(args, logger)
    if sub == "topics":
        return run_kg_topics(args, logger)
    logger.error("Unknown kg subcommand: %s", sub)
    return 1
