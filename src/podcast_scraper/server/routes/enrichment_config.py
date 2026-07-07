"""GET/PUT enrichment config slice + JSON Schema + provider-type catalogue.

Routes:

* ``GET  /api/enrichment/config`` — resolved view of the
  ``enrichment:`` block (profile YAML + operator YAML deep-merged) plus
  the raw operator-side override so the UI knows what's inherited vs
  what's been customised.
* ``PUT  /api/enrichment/config`` — write the operator-side
  ``enrichment:`` block to ``<corpus>/viewer_operator.yaml`` after
  validating against the JSON Schema. Atomic; preserves unrelated
  keys.
* ``GET  /api/enrichment/config/schema`` — full JSON Schema for the
  ``enrichment:`` block composed from
  ``config/schema/enrichment.schema.json`` + each enricher's manifest
  ``config_schema`` fragment + each provider type's ``params_schema``
  fragment. Used by the viewer Configuration → Enrichment editor for
  form generation.
* ``GET  /api/enrichment/provider-types`` — registry catalogue
  grouped by protocol (EmbeddingProvider / NliScorer / ...). UI
  populates per-row provider dropdowns from this.

Companion to :mod:`podcast_scraper.server.routes.enrichment` (which
serves the operator-facing status / health / metrics) and
:mod:`podcast_scraper.server.routes.corpus_enrichments` (which
serves on-disk envelopes). This module is the *write* surface.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from podcast_scraper.enrichment.config_schema import (
    ConfigSchemaError,
    validate_enrichment_block,
)
from podcast_scraper.enrichment.enrichers import register_deterministic_enrichers
from podcast_scraper.enrichment.provider_types import get_global_registry
from podcast_scraper.enrichment.registry import EnricherRegistry
from podcast_scraper.server.atomic_write import atomic_write_text
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.profile_presets import packaged_profile_contents

router = APIRouter(tags=["enrichment-config"])


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------


class EnrichmentConfigGetResponse(BaseModel):
    """``GET /api/enrichment/config`` payload."""

    corpus_path: str
    profile: str | None = Field(
        default=None,
        description="Resolved profile name driving the base enrichment set.",
    )
    profile_block: dict[str, Any] = Field(
        default_factory=dict,
        description="Enrichment block from the named profile YAML (the base).",
    )
    operator_block: dict[str, Any] = Field(
        default_factory=dict,
        description="Enrichment block from viewer_operator.yaml (the override).",
    )
    resolved_block: dict[str, Any] = Field(
        default_factory=dict,
        description="Deep-merge of profile + operator. What the executor would run.",
    )


class EnrichmentConfigPutBody(BaseModel):
    """``PUT /api/enrichment/config`` request body."""

    enrichment_block: dict[str, Any] = Field(
        description="Operator-side ``enrichment:`` block to persist.",
    )


class ProviderTypeInfo(BaseModel):
    """One provider type's UI-relevant metadata."""

    name: str
    protocol: str
    description: str
    params_schema: dict[str, Any]


class ProviderTypesResponse(BaseModel):
    """``GET /api/enrichment/provider-types`` payload."""

    by_protocol: dict[str, list[ProviderTypeInfo]]


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *over* into *base*. ``over`` wins per leaf key.

    Non-dict values replace wholesale. Lists are NOT deep-merged —
    they're replaced (e.g. a future ``enrichment.priority_order: [...]``
    operator override).
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _viewer_operator_yaml_path(corpus_root: Path) -> Path:
    return corpus_root / "viewer_operator.yaml"


def _read_operator_yaml(corpus_root: Path) -> dict[str, Any]:
    path = _viewer_operator_yaml_path(corpus_root)
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _read_profile_block(profile: str | None) -> dict[str, Any]:
    """Load the named profile YAML's ``enrichment:`` block (or ``{}``)."""
    if not profile:
        return {}
    contents = packaged_profile_contents()
    raw = contents.get(profile)
    if not isinstance(raw, str):
        return {}
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    block = parsed.get("enrichment")
    return block if isinstance(block, dict) else {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/enrichment/config", response_model=EnrichmentConfigGetResponse)
async def get_enrichment_config(
    request: Request,
    path: str = Query(
        ..., description="Corpus root (authorizes request; must resolve under anchor)."
    ),
) -> EnrichmentConfigGetResponse:
    """Resolved enrichment config view: profile block + operator override + merged."""
    anchor = getattr(request.app.state, "output_dir", None)
    corpus_root = resolve_corpus_path_param(path, anchor, must_be_dir=False)
    operator_yaml = _read_operator_yaml(corpus_root)
    raw_block = operator_yaml.get("enrichment")
    operator_block: dict[str, Any] = raw_block if isinstance(raw_block, dict) else {}
    raw_profile = operator_yaml.get("profile")
    profile: str | None = raw_profile if isinstance(raw_profile, str) else None
    profile_block = _read_profile_block(profile)
    resolved = _deep_merge(profile_block, operator_block)
    return EnrichmentConfigGetResponse(
        corpus_path=str(corpus_root),
        profile=profile,
        profile_block=profile_block,
        operator_block=operator_block,
        resolved_block=resolved,
    )


@router.put("/enrichment/config", response_model=EnrichmentConfigGetResponse)
async def put_enrichment_config(
    request: Request,
    body: EnrichmentConfigPutBody,
    path: str = Query(
        ..., description="Corpus root (authorizes request; must resolve under anchor)."
    ),
) -> EnrichmentConfigGetResponse:
    """Validate + persist the operator-side ``enrichment:`` block.

    Atomic: rewrites the whole ``viewer_operator.yaml`` with the new
    enrichment block while preserving every unrelated top-level key.
    """
    anchor = getattr(request.app.state, "output_dir", None)
    corpus_root = resolve_corpus_path_param(path, anchor, must_be_dir=False)
    block = body.enrichment_block
    # Validate against the BASE schema first (catches structural mistakes
    # like `enabled: "yes please"`), then against the COMPOSED schema
    # (catches unknown provider types, missing required provider params,
    # typo'd knob names, providers on deterministic enrichers).
    try:
        validate_enrichment_block(block)
    except ConfigSchemaError as exc:
        raise HTTPException(status_code=400, detail=f"invalid enrichment block: {exc}") from exc
    try:
        _validate_against_composed_schema(block)
    except ConfigSchemaError as exc:
        raise HTTPException(status_code=400, detail=f"invalid enrichment block: {exc}") from exc
    operator_yaml = _read_operator_yaml(corpus_root)
    operator_yaml["enrichment"] = block
    serialised = yaml.safe_dump(operator_yaml, sort_keys=False, default_flow_style=False)
    atomic_write_text(_viewer_operator_yaml_path(corpus_root), serialised)
    # Return the fresh resolved view (consistent with GET).
    return await get_enrichment_config(request=request, path=path)


def _build_composed_schema() -> dict[str, Any]:
    """Compose base + per-enricher fragments + provider params. Shared by
    the /schema GET route and PUT-time validation so the operator and
    the server agree on exactly what's accepted.
    """
    from podcast_scraper.enrichment.config_schema import load_schema
    from podcast_scraper.enrichment.enrichers.stance_timeline import StanceTimelineEnricher
    from podcast_scraper.enrichment.enrichers.topic_consensus import (
        TopicConsensusEnricher,
    )
    from podcast_scraper.enrichment.enrichers.topic_similarity import (
        TopicSimilarityEnricher,
    )

    base = load_schema()
    enricher_blocks: dict[str, Any] = {}
    reg = EnricherRegistry()
    register_deterministic_enrichers(reg)
    for eid in reg.all_ids():
        m = reg.get(eid).manifest
        enricher_blocks[m.id] = _per_enricher_schema(m)
    for cls in (TopicSimilarityEnricher, TopicConsensusEnricher, StanceTimelineEnricher):
        m = cls.manifest  # type: ignore[attr-defined]
        enricher_blocks[m.id] = _per_enricher_schema(m)
    out = copy.deepcopy(base)
    enrichers_prop = out.setdefault("properties", {}).setdefault("enrichers", {})
    enrichers_prop["properties"] = enricher_blocks
    # Tighten: when the YAML names an enricher key not in the registry,
    # validation should reject it. The composed schema's per-enricher
    # blocks each set additionalProperties on the BLOCK; the outer
    # ``enrichers`` map keeps additionalProperties open so legacy
    # operator YAMLs with unknown ids don't 400 the route, but the schema
    # GET surface still teaches the canonical set via ``properties``.
    return out


def _validate_against_composed_schema(block: dict[str, Any]) -> None:
    """Validate *block* against the composed schema. Raises
    :class:`ConfigSchemaError` on failure.

    Catches everything the base schema misses: unknown provider types,
    missing required provider params (e.g.
    ``sentence_transformer_local`` requires ``model``), typo'd knob
    names (``alphaa`` instead of ``alpha``), and provider blocks on
    deterministic enrichers (their composed fragment doesn't include a
    ``provider`` property).
    """
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover — jsonschema is in [dev]
        return
    composed = _build_composed_schema()
    try:
        jsonschema.validate(block, composed)
    except jsonschema.ValidationError as exc:
        # Format: "{path}: {message}" — easier to act on than the default.
        path = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        raise ConfigSchemaError(f"{path}: {exc.message}") from exc


@router.get("/enrichment/config/schema")
async def get_enrichment_config_schema() -> dict[str, Any]:
    """Full JSON Schema for the ``enrichment:`` block (data-driven UI).

    Composed from:
      - the base ``config/schema/enrichment.schema.json``
      - each enricher's ``manifest.config_schema`` (composed under
        ``enrichers.<id>``'s ``properties``)
      - each provider type's ``params_schema`` (referenced via
        ``oneOf`` so the UI can pick the right form per type)
    """
    try:
        return _build_composed_schema()
    except ConfigSchemaError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class EnricherAdmissionRow(BaseModel):
    """One enricher's accuracy-gate admission status."""

    id: str
    tier: str
    scope: str
    has_gate: bool  # True when the manifest declares an accuracy_gate
    promoted: bool  # True when admitted to the registry (→ profiles)
    reason: str  # human-facing, e.g. "gated: precision 0.00 < 0.50"


class EnrichmentAdmissionResponse(BaseModel):
    """``GET /api/enrichment/config/admission`` payload."""

    enrichers: list[EnricherAdmissionRow]


@router.get("/enrichment/config/admission", response_model=EnrichmentAdmissionResponse)
async def get_enrichment_admission() -> EnrichmentAdmissionResponse:
    """Per-enricher accuracy-gate admission status (the UI leg of the cascade).

    Surfaces the ``data/eval`` → gate decision (promote/reject + reason) for
    every known enricher, so the Configuration → Enrichment editor can show
    *why* an enricher is off ("gated: precision 0.00 < 0.50") rather than just
    its silent absence. Mirrors how provider gate status is surfaced. The gate
    decision is profile-independent (a failing eval gates the enricher
    everywhere), so this reports one row per known enricher.
    """
    from podcast_scraper.enrichment.eval.admission import (
        admit_enrichers,
        known_enricher_manifests,
    )

    manifests = known_enricher_manifests()
    result = admit_enrichers(list(manifests))
    rows = [
        EnricherAdmissionRow(
            id=eid,
            tier=m.tier.value,
            scope=m.scope.value,
            has_gate=m.accuracy_gate is not None,
            promoted=result.decisions[eid].promoted,
            reason=result.decisions[eid].reason,
        )
        for eid, m in manifests.items()
    ]
    return EnrichmentAdmissionResponse(enrichers=rows)


def _per_enricher_schema(manifest: Any) -> dict[str, Any]:
    """Build the per-enricher schema fragment composed of base keys +
    config_schema knobs + provider block (oneOf of registered types)."""
    base_props: dict[str, Any] = {
        "enabled": {
            "type": "boolean",
            "description": (
                "Defaults true when omitted. Set false to opt out without " "removing the block."
            ),
        },
        "opt_in": {"type": "boolean"},
        "max_cost_usd_per_run": {"type": "number", "minimum": 0},
        "expected_duration_s": {"type": "integer", "minimum": 1},
    }
    # ``additionalProperties: false`` catches typo'd knob names + provider
    # blocks on deterministic enrichers (manifest.provider_requirement is
    # None → ``provider`` isn't added to base_props → schema rejects it).
    block: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": base_props,
    }
    cs = getattr(manifest, "config_schema", None)
    if isinstance(cs, dict) and isinstance(cs.get("properties"), dict):
        for k, v in cs["properties"].items():
            base_props[k] = v
    pr = getattr(manifest, "provider_requirement", None)
    if pr is not None:
        provider_types = get_global_registry().list_for_protocol(pr.protocol)
        oneof: list[dict[str, Any]] = []
        for pt in provider_types:
            entry_props: dict[str, Any] = {
                "type": {"const": pt.name, "description": pt.description},
            }
            required: list[str] = ["type"]
            params = pt.params_schema or {}
            params_props = params.get("properties")
            if isinstance(params_props, dict):
                for pk, pv in params_props.items():
                    entry_props[pk] = pv
                params_required = params.get("required")
                if isinstance(params_required, list):
                    required = sorted({*required, *params_required})
            entry: dict[str, Any] = {
                "type": "object",
                "required": required,
                "properties": entry_props,
                "additionalProperties": True,
            }
            oneof.append(entry)
        base_props["provider"] = {
            "description": (
                f"Provider injection for protocol {pr.protocol!r}. " f"{pr.description}"
            ),
            "oneOf": oneof or [{"type": "object"}],
        }
    return block


@router.get("/enrichment/provider-types", response_model=ProviderTypesResponse)
async def get_provider_types() -> ProviderTypesResponse:
    """Catalogue of registered provider types grouped by protocol."""
    reg = get_global_registry()
    by_protocol: dict[str, list[ProviderTypeInfo]] = {}
    for pt in reg.all_types():
        by_protocol.setdefault(pt.protocol, []).append(
            ProviderTypeInfo(
                name=pt.name,
                protocol=pt.protocol,
                description=pt.description,
                params_schema=pt.params_schema,
            )
        )
    for protocol in by_protocol:
        by_protocol[protocol].sort(key=lambda t: t.name)
    return ProviderTypesResponse(by_protocol=by_protocol)


__all__ = ["router"]
