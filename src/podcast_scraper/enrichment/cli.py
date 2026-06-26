"""``podcast enrich`` CLI surface.

Operator entry point for one-shot / backfill / dev-iteration runs.
Production / cron / viewer-triggered runs go through the jobs API
(``POST /api/jobs/enrichment``) instead — see ``server/jobs.py``
``COMMAND_ENRICHMENT`` integration in sub-6.

Usage:

    podcast enrich --output-dir <corpus>
                   [--only id,id]
                   [--skip id,id]
                   [--corpus-only]
                   [--re-enable id]
                   [--config <yaml>]

The CLI:

1. Loads the operator YAML and validates the ``enrichment:`` block
   against ``config/schema/enrichment.schema.json``.
2. Builds an EnricherSet from the YAML.
3. Reads the registry (currently empty in chunk 1; chunk 2+
   enrichers register at import time).
4. For ``--re-enable``: edits health and exits (no run).
5. Otherwise: scans the corpus for episode bundles and invokes
   ``EnrichmentExecutor.run(...)``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.executor import (
    EnrichmentExecutor,
    EnrichmentRunResult,
    ExecutorOptions,
)
from podcast_scraper.enrichment.health import HealthRegistry
from podcast_scraper.enrichment.paths import enrichment_health_path
from podcast_scraper.enrichment.protocol import EnricherSet
from podcast_scraper.enrichment.registry import EnricherRegistry

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``podcast enrich`` arg parser.

    Returned bare so callers can wrap it into a sub-parser of the main
    ``podcast`` CLI without re-implementing the flags.
    """
    parser = argparse.ArgumentParser(
        prog="podcast enrich",
        description="Run enrichment-layer pass over a corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Corpus output directory (the same one used by ``podcast`` proper).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated enricher ids to include (defaults to the active set).",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default=None,
        help="Comma-separated enricher ids to skip.",
    )
    parser.add_argument(
        "--corpus-only",
        action="store_true",
        help="Skip the episode-scope phase; run only corpus-scope enrichers.",
    )
    parser.add_argument(
        "--re-enable",
        type=str,
        default=None,
        help=(
            "Re-enable an auto-disabled enricher and exit (no run). "
            "Resets consecutive_failures + clears cooldown + stamps reason."
        ),
    )
    parser.add_argument(
        "--re-enable-reason",
        type=str,
        default="manual re_enable via CLI",
        help="Audit-trail reason recorded with --re-enable.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to operator YAML (the ``enrichment:`` block is validated; "
            "other keys are ignored). Defaults to the corpus's stored config."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def parse_id_list(raw: str | None) -> list[str] | None:
    """Convert a comma-separated string to a list (or None).

    Whitespace trimmed; empty entries dropped.
    """
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    cleaned = [p for p in parts if p]
    return cleaned or None


def re_enable_enricher(
    *,
    corpus_root: Path,
    enricher_id: str,
    reason: str,
) -> dict[str, Any]:
    """Operator manual recovery — update health, persist, return record.

    Loads the existing health file (idempotent on missing) so partial
    state isn't clobbered.
    """
    health = HealthRegistry(corpus_root)
    health.load()
    record = health.re_enable(enricher_id, reason=reason, clear_cooldown=True)
    health.save()
    logger.info(
        "enrichment: re-enabled %r (cleared cooldown; counter reset). " "Health file: %s",
        enricher_id,
        enrichment_health_path(corpus_root),
    )
    return {
        "enricher_id": enricher_id,
        "auto_disabled": record.auto_disabled,
        "consecutive_failures": record.consecutive_failures,
        "auto_disabled_reason": record.auto_disabled_reason,
        "circuit_state": record.circuit_state,
        "cooldown_until": record.cooldown_until,
    }


async def run_cli(args: argparse.Namespace) -> int:
    """Async entry point shared by ``main()`` and tests."""
    corpus_root: Path = args.output_dir
    if not corpus_root.is_dir():
        logger.error("enrichment: --output-dir does not exist: %s", corpus_root)
        return 2

    if args.re_enable:
        record = re_enable_enricher(
            corpus_root=corpus_root,
            enricher_id=args.re_enable,
            reason=args.re_enable_reason,
        )
        print(f"re-enabled: {record}")
        return 0

    # Build the enricher set — in chunk 1 the registry is empty; the
    # CLI exits cleanly as a no-op so operators can wire profile-preset
    # configs and the executor without errors.
    registry = EnricherRegistry()
    enricher_set = build_enricher_set_from_yaml(args.config)
    executor = EnrichmentExecutor(
        corpus_root=corpus_root,
        registry=registry,
        enricher_set=enricher_set,
    )

    options = ExecutorOptions(
        only=parse_id_list(args.only),
        skip=parse_id_list(args.skip),
        corpus_only=bool(args.corpus_only),
    )
    result: EnrichmentRunResult = await executor.run(
        episode_bundles=[],  # chunk 1 ships no episode-scoping; sub-6 wires
        options=options,
    )
    print(
        f"enrichment: run_id={result.run_id} status={result.status} "
        f"duration_ms={result.duration_ms}"
    )
    return 0 if result.status in ("ok", "skipped") else 1


def build_enricher_set_from_yaml(config_path: Path | None) -> EnricherSet:
    """Read the ``enrichment:`` block from YAML (if present).

    Schema validation is performed by ``config_schema.validate_enrichment_block``
    (defined alongside the JSON Schema). Missing file → empty set
    (the no-op path runs cleanly).
    """
    if config_path is None or not config_path.is_file():
        return EnricherSet()
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("enrichment: PyYAML not installed; --config will be ignored")
        return EnricherSet()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("enrichment: cannot read --config (%s): %s", config_path, exc)
        return EnricherSet()
    if not isinstance(raw, dict):
        return EnricherSet()
    block = raw.get("enrichment") or {}
    if not isinstance(block, dict):
        return EnricherSet()

    # Schema validation (deferred import to avoid hard jsonschema dep
    # for envs that never call enrich).
    from podcast_scraper.enrichment.config_schema import (
        ConfigSchemaError,
        validate_enrichment_block,
    )

    try:
        validate_enrichment_block(block)
    except ConfigSchemaError as exc:
        logger.error("enrichment: invalid 'enrichment:' block in config: %s", exc)
        raise SystemExit(2) from exc

    enrichers_cfg: dict[str, dict[str, Any]] = block.get("enrichers") or {}
    enabled: list[str] = []
    per_enricher_config: dict[str, dict[str, Any]] = {}
    opt_in_flags: dict[str, bool] = {}
    for eid, cfg in enrichers_cfg.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("enabled", False):
            enabled.append(eid)
        per_enricher_config[eid] = cfg
        if "opt_in" in cfg:
            opt_in_flags[eid] = bool(cfg["opt_in"])
    return EnricherSet(
        enabled_enrichers=enabled,
        per_enricher_config=per_enricher_config,
        opt_in_flags=opt_in_flags,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return asyncio.run(run_cli(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
