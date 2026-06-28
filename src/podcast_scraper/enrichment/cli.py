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
3. Registers the six deterministic enrichers via
   :func:`enrichers.register_deterministic_enrichers`. ML / external
   enrichers (topic_similarity, nli_contradiction) need provider /
   scorer wiring and are registered by callers that supply it.
4. For ``--re-enable``: edits health and exits (no run).
5. Otherwise: discovers episode bundles via
   :func:`paths.discover_episode_bundles` (latest-run-per-feed dedup)
   and invokes ``EnrichmentExecutor.run(...)``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers import register_deterministic_enrichers
from podcast_scraper.enrichment.executor import (
    EnrichmentExecutor,
    EnrichmentRunResult,
    ExecutorOptions,
)
from podcast_scraper.enrichment.health import HealthRegistry
from podcast_scraper.enrichment.paths import discover_episode_bundles, enrichment_health_path
from podcast_scraper.enrichment.profile_sets import (
    apply_cli_overrides,
    enricher_set_for_profile,
)
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
    # RFC-088 chunk 7: profile-preset overrides.
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "Profile preset name (matches config/profiles/*.yaml). When set, "
            "the EnricherSet is derived from the chunk-7 matrix unless --only "
            "/ --skip / --no-enrichers override."
        ),
    )
    parser.add_argument(
        "--enrichers",
        type=str,
        default=None,
        help=(
            "Comma-separated enricher ids to force-include (alias for --only "
            "kept for parity with the plan body)."
        ),
    )
    parser.add_argument(
        "--no-enrichers",
        action="store_true",
        help=(
            "Disable every enricher for this run (still emits run.skipped + "
            "run.completed events so the JSONL audit trail stays whole)."
        ),
    )
    parser.add_argument(
        "--opt-in",
        type=str,
        default=None,
        help=(
            "Comma-separated enricher ids to mark as operator-opt-in (required "
            "by enrichers with manifest.requires_opt_in=True). Layers on top of "
            "the profile's defaults."
        ),
    )
    parser.add_argument(
        "--with-ml",
        action="store_true",
        help=(
            "Wire ML / embedding / NLI enrichers (topic_similarity, "
            "nli_contradiction, ...) from the EnricherSet's "
            "per_enricher_config[id].provider blocks. Without this flag, "
            "the deterministic enrichers run; ML enrichers are skipped "
            "with a hinted WARNING. Operators running locally with the "
            "[ml] extra installed turn this on; the workflow path passes "
            "it automatically when the resolved EnricherSet needs ML."
        ),
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

    # Build the enricher set. Resolution model (Shape B / RFC-088 v2):
    #   1. The --profile preset gives the BASE EnricherSet (the
    #      hardcoded matrix in profile_sets.enricher_set_for_profile).
    #   2. The operator YAML's ``enrichment.enrichers:`` dict is the
    #      OVERRIDE layer, read in Shape B (block-present = enabled,
    #      explicit ``enabled: false`` = opt-out). Knobs + provider
    #      config merge per-key over the profile's defaults.
    #   3. opt_in_flags merge per-key.
    #   4. CLI flags layer on top: --no-enrichers / --enrichers (alias
    #      for --only) / --only / --skip / --opt-in.
    registry = EnricherRegistry()
    # Deterministic enrichers are always registered — they have no external
    # dependencies. Tier=DETERMINISTIC manifests gate their own activation
    # via the EnricherSet's enabled list; presence in the registry is
    # purely "available". Tier=ML / EXTERNAL enrichers (topic_similarity,
    # nli_contradiction) need provider/scorer wiring and are registered
    # by the call site that supplies that wiring.
    register_deterministic_enrichers(registry)
    yaml_set = build_enricher_set_from_yaml(args.config)
    profile_set = enricher_set_for_profile(args.profile)
    base_enabled = list(yaml_set.enabled_enrichers) or list(profile_set.enabled_enrichers)
    base_set = EnricherSet(
        enabled_enrichers=base_enabled,
        per_enricher_config={
            **profile_set.per_enricher_config,
            **yaml_set.per_enricher_config,
        },
        opt_in_flags={**profile_set.opt_in_flags, **yaml_set.opt_in_flags},
    )
    only_arg = args.enrichers or args.only
    enricher_set = apply_cli_overrides(
        base_set,
        only=parse_id_list(only_arg),
        skip=parse_id_list(args.skip),
        no_enrichers=bool(args.no_enrichers),
        extra_opt_in=parse_id_list(args.opt_in),
    )
    # --with-ml: register ML / embedding / NLI enrichers from the
    # provider-type registry, using each enricher's per_enricher_config
    # ``provider`` block to pick the type + params. Without this flag,
    # the registry.list_enabled() path emits a hinted WARNING for any
    # un-registered ML enricher (see registry.py:_PROVIDER_WIRING_HINT)
    # — operators see exactly why those enrichers got skipped.
    if args.with_ml:
        from podcast_scraper.enrichment.ml_wiring import register_ml_enrichers

        register_ml_enrichers(registry, enricher_set)
    executor = EnrichmentExecutor(
        corpus_root=corpus_root,
        registry=registry,
        enricher_set=enricher_set,
    )

    options = ExecutorOptions(
        only=enricher_set.enabled_enrichers or None,
        skip=parse_id_list(args.skip),
        corpus_only=bool(args.corpus_only),
        profile=args.profile,
    )
    # Discover episode bundles unless --corpus-only is set (corpus-scope
    # enrichers still need them as ``all_bundles``, but --corpus-only
    # callers may want to skip the walk if they've already scoped down).
    episode_bundles = discover_episode_bundles(corpus_root)
    logger.info(
        "enrichment: discovered %d episode bundle(s) under %s",
        len(episode_bundles),
        corpus_root,
    )
    result: EnrichmentRunResult = await executor.run(
        episode_bundles=episode_bundles,
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

    # Shape B (RFC-088 config v2): the ``enrichers:`` block is a dict
    # keyed by enricher id. Presence of a block IS the enable; explicit
    # ``enabled: false`` opts an enricher out. This matches the YAML
    # operator mental model (one block per enricher, all info in one
    # place) and the UI's per-row v-model binding.
    #
    # Legacy support: the dict-of-blocks shape with explicit
    # ``enabled: true`` keys still works (the implicit-default rule
    # makes both readings equivalent). The list-of-ids shape used by
    # built-in profiles is handled by ``enricher_set_for_profile``,
    # not by this YAML reader.
    enrichers_raw = block.get("enrichers")
    enabled: list[str] = []
    per_enricher_config: dict[str, dict[str, Any]] = {}
    opt_in_flags: dict[str, bool] = {}
    if not isinstance(enrichers_raw, dict):
        # Defensive: a list shape leaked into operator YAML by accident
        # / legacy migration. Honour it as "enable these, no config",
        # but the schema rejects it on validation so this branch is
        # rarely hit.
        if isinstance(enrichers_raw, list):
            enabled = [eid for eid in enrichers_raw if isinstance(eid, str)]
            return EnricherSet(enabled_enrichers=enabled)
        return EnricherSet()
    for eid, cfg in enrichers_raw.items():
        if not isinstance(cfg, dict):
            # Bare ``enricher_id: null`` / ``enricher_id: true`` etc.
            # Treat as enabled with no config (presence-is-enabled).
            enabled.append(eid)
            continue
        # Implicit-enabled-default: block present + no explicit
        # ``enabled: false`` → on. Explicit ``enabled: false`` → off.
        if cfg.get("enabled", True):
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
