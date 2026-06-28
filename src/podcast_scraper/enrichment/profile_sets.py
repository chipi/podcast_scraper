"""Profile-preset → EnricherSet matrix (RFC-088 chunk 7).

Maps each operator profile to the enricher set that should run under it.
Decoupled from ``providers.ml.model_registry.ProfilePreset`` (which is
ML-stage compositions, a different concern) — instead this module owns
the small per-profile lookup table the executor / CLI consume.

Source of truth ([[feedback_profiles_are_source_of_truth]]):

- The profile names here MUST match the YAMLs under ``config/profiles/``.
- ``test_default`` runs no enrichers (CI isolation).
- ``airgapped_thin`` runs only the 6 deterministic enrichers.
- ``airgapped`` adds ``topic_similarity`` (local CPU embedding).
- ``cloud_thin`` adds ``nli_contradiction`` on top (CPU NLI, CI-safe
  with the FixedNliScorer fixture; the real DeBERTa loads only when
  the operator runs locally).
- ``cloud_balanced`` / ``cloud_quality`` get the full set.
- ``dev`` / ``prod`` / ``local`` / DGX variants mirror their parent
  level — they don't carry their own enricher policies yet.

A drift test (``test_profile_sets.py``) asserts every preset YAML under
``config/profiles/*.yaml`` has a row here, so adding a new profile
without registering its enricher set fails CI loudly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers import ALL_DETERMINISTIC_ENRICHER_IDS
from podcast_scraper.enrichment.protocol import EnricherSet

logger = logging.getLogger(__name__)


def _read_profile_yaml_enrichers(profile: str) -> dict[str, dict[str, Any]]:
    """Read per-enricher knob/provider configs from a profile YAML.

    Profile YAMLs (under ``config/profiles/<profile>.yaml``) carry the
    canonical Shape B ``enrichment.enrichers:`` dict — empty blocks
    (``temporal_velocity: {}``) by default, but operators can ship
    profile-side knob defaults (``temporal_velocity: {alpha: 0.7}``)
    or default provider blocks for the ML enrichers
    (``topic_similarity: {provider: {...}}``) without having to
    duplicate them in every corpus's ``viewer_operator.yaml``.

    Returns ``{}`` on missing / unparsable / non-dict block. Schema
    validation lives on the operator path (the v2 composed schema
    is the strict gate); this reader is permissive so a slightly
    malformed profile YAML doesn't make the matrix call collapse.
    """
    profiles_dir = Path(__file__).resolve().parents[3] / "config" / "profiles"
    yaml_path = profiles_dir / f"{profile}.yaml"
    if not yaml_path.is_file():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return {}
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(raw, dict):
        return {}
    block = raw.get("enrichment")
    if not isinstance(block, dict):
        return {}
    enrichers = block.get("enrichers")
    if not isinstance(enrichers, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for eid, cfg in enrichers.items():
        if isinstance(cfg, dict):
            out[eid] = cfg
    return out


# Profile presets that currently get the no-enricher set (mostly the
# leaf "freeze" profiles + the test_default).
_NO_ENRICHERS_PROFILES: frozenset[str] = frozenset(
    {
        "test_default",
        "eval_default",
        # profile_freeze.example is config-only — never a real run.
        "profile_freeze.example",
        # Pre-prod is for local Whisper testing — keep enrichment off so
        # operator-side benchmarking is uncoupled.
        "preprod_local_whisper",
    }
)


# Profile preset → enricher set membership. Built lazily so callers
# can monkey-patch in tests without import-order surprises.
def _deterministic_only() -> list[str]:
    return list(ALL_DETERMINISTIC_ENRICHER_IDS)


def _with_topic_similarity() -> list[str]:
    return [*ALL_DETERMINISTIC_ENRICHER_IDS, "topic_similarity"]


def _with_nli_too() -> list[str]:
    return [*ALL_DETERMINISTIC_ENRICHER_IDS, "topic_similarity", "nli_contradiction"]


# When the operator opts in to nli_contradiction this is the flag that
# satisfies the LLM-tier double-opt-in for ``requires_opt_in=True``
# enrichers. nli_contradiction itself is CPU-local so it doesn't carry
# the flag, but future LLM query enrichers will.
_DEFAULT_OPT_IN_FLAGS: dict[str, bool] = {}


def enricher_set_for_profile(profile: str | None) -> EnricherSet:
    """Return the canonical EnricherSet for *profile*.

    Unknown / None profiles return the empty no-enricher set — safe
    default that matches ``test_default``. The profile YAML's
    ``enrichment.enrichers:`` dict is read for per-enricher knob +
    provider config so operators can ship profile-side defaults
    (e.g. ``temporal_velocity: {alpha: 0.7}``) without per-corpus
    YAML duplication. The Python matrix below remains the source of
    truth for which ids are ENABLED — the YAML supplies CONFIG only.
    """
    if not profile or profile.strip() in _NO_ENRICHERS_PROFILES:
        return EnricherSet()
    name = profile.strip()
    per_enricher_config = _read_profile_yaml_enrichers(name)

    if name == "airgapped_thin":
        return EnricherSet(
            enabled_enrichers=_deterministic_only(),
            per_enricher_config=per_enricher_config,
        )

    if name in ("airgapped",):
        return EnricherSet(
            enabled_enrichers=_with_topic_similarity(),
            per_enricher_config=per_enricher_config,
            opt_in_flags=dict(_DEFAULT_OPT_IN_FLAGS),
        )

    if name in ("cloud_thin", "cloud_balanced", "cloud_quality"):
        return EnricherSet(
            enabled_enrichers=_with_nli_too(),
            per_enricher_config=per_enricher_config,
            opt_in_flags=dict(_DEFAULT_OPT_IN_FLAGS),
        )

    # Hybrid / DGX / local profiles inherit the full cloud set —
    # they all have a real LLM somewhere, so the gating constraint
    # is the same. Note: only profile names with a real YAML under
    # config/profiles/ are matched here ('prod' was dropped because
    # there is no config/profiles/prod.yaml; the production-shaped
    # profiles are prod_dgx_balanced / prod_dgx_full_with_fallback).
    if (
        name.startswith("local_dgx_")
        or name.startswith("prod_dgx_")
        or name
        in (
            "cloud_with_dgx_primary",
            "dev",
            "local",
        )
    ):
        return EnricherSet(
            enabled_enrichers=_with_nli_too(),
            per_enricher_config=per_enricher_config,
            opt_in_flags=dict(_DEFAULT_OPT_IN_FLAGS),
        )

    # Unknown profile — be conservative: no enrichers. Log a WARNING so
    # operators see this in their server logs (silent empty set masked
    # typo'd profile names in pre-A9 follow-up code).
    logger.warning(
        "enrichment.profile_sets: unknown profile %r; defaulting to empty enricher "
        "set. Add a branch in enricher_set_for_profile() or use one of the known "
        "profile names.",
        profile,
    )
    return EnricherSet()


def discover_profile_yaml_names(profiles_dir: Path | None = None) -> list[str]:
    """Return every profile name with a YAML under ``config/profiles/``.

    Skips ``*.example.yaml`` and the ``freeze/`` subdirectory (those are
    config-only fragments, not selectable profile presets).
    """
    root = profiles_dir or Path(__file__).resolve().parents[3] / "config" / "profiles"
    names: list[str] = []
    if not root.is_dir():
        return names
    for entry in sorted(root.glob("*.yaml")):
        if entry.name.endswith(".example.yaml"):
            continue
        names.append(entry.stem)
    return names


class UnknownOptInError(ValueError):
    """``--opt-in`` named an enricher that isn't in the active set."""


def apply_cli_overrides(
    base: EnricherSet,
    *,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    no_enrichers: bool = False,
    extra_opt_in: list[str] | None = None,
) -> EnricherSet:
    """Apply CLI / API overrides on top of the profile-derived set.

    Precedence (highest first):
      1. ``no_enrichers=True`` → empty set, ignore everything else.
      2. ``only=[...]`` restricts to that subset (and ``skip`` is then
         a finer filter over what's left). Empty list is a no-op (no
         filtering applied).
      3. ``skip=[...]`` removes those ids from the base set. Empty list
         is a no-op.
      4. ``extra_opt_in=[...]`` adds opt-in flags on top of the base.
         Raises :class:`UnknownOptInError` if an id is not in the
         resulting enabled set — opting in to an enricher that won't run
         is almost always a typo (e.g. ``--opt-in nly_contradiction``).
    """
    if no_enrichers:
        return EnricherSet()
    enabled = list(base.enabled_enrichers)
    if only:
        if not enabled:
            # No base set (no YAML, no profile) — treat ``only`` as the
            # full list to run. Matches the operator mental model: typing
            # ``--enrichers a,b,c`` should RUN a,b,c, not silently no-op
            # because the empty profile filter swallowed them.
            enabled = list(only)
        else:
            keep = set(only)
            enabled = [e for e in enabled if e in keep]
    if skip:
        drop = set(skip)
        enabled = [e for e in enabled if e not in drop]
    opt_in = dict(base.opt_in_flags)
    if extra_opt_in:
        unknown = [eid for eid in extra_opt_in if eid not in set(enabled)]
        if unknown:
            raise UnknownOptInError(
                f"--opt-in id(s) not in active enricher set: {unknown}. "
                f"Active set: {sorted(enabled)}"
            )
        for eid in extra_opt_in:
            opt_in[eid] = True
    return EnricherSet(
        enabled_enrichers=enabled,
        per_enricher_config=dict(base.per_enricher_config),
        opt_in_flags=opt_in,
    )


__all__ = [
    "UnknownOptInError",
    "apply_cli_overrides",
    "discover_profile_yaml_names",
    "enricher_set_for_profile",
]


_ = Any  # quell ruff/flake8 unused-import suspicion on Any
