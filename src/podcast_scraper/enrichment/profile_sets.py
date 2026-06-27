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

from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers import ALL_DETERMINISTIC_ENRICHER_IDS
from podcast_scraper.enrichment.protocol import EnricherSet

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
    default that matches ``test_default``.
    """
    if not profile or profile.strip() in _NO_ENRICHERS_PROFILES:
        return EnricherSet()
    name = profile.strip()

    if name == "airgapped_thin":
        return EnricherSet(enabled_enrichers=_deterministic_only())

    if name in ("airgapped",):
        return EnricherSet(
            enabled_enrichers=_with_topic_similarity(),
            opt_in_flags=dict(_DEFAULT_OPT_IN_FLAGS),
        )

    if name in ("cloud_thin", "cloud_balanced", "cloud_quality"):
        return EnricherSet(
            enabled_enrichers=_with_nli_too(),
            opt_in_flags=dict(_DEFAULT_OPT_IN_FLAGS),
        )

    # Hybrid / DGX / local profiles inherit the full cloud set —
    # they all have a real LLM somewhere, so the gating constraint
    # is the same.
    if (
        name.startswith("local_dgx_")
        or name.startswith("prod_dgx_")
        or name
        in (
            "cloud_with_dgx_primary",
            "dev",
            "prod",
            "local",
        )
    ):
        return EnricherSet(
            enabled_enrichers=_with_nli_too(),
            opt_in_flags=dict(_DEFAULT_OPT_IN_FLAGS),
        )

    # Unknown profile — be conservative: no enrichers.
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
         a finer filter over what's left).
      3. ``skip=[...]`` removes those ids from the base set.
      4. ``extra_opt_in=[...]`` adds opt-in flags on top of the base.
    """
    if no_enrichers:
        return EnricherSet()
    enabled = list(base.enabled_enrichers)
    if only:
        keep = set(only)
        enabled = [e for e in enabled if e in keep]
    if skip:
        drop = set(skip)
        enabled = [e for e in enabled if e not in drop]
    opt_in = dict(base.opt_in_flags)
    if extra_opt_in:
        for eid in extra_opt_in:
            opt_in[eid] = True
    return EnricherSet(
        enabled_enrichers=enabled,
        per_enricher_config=dict(base.per_enricher_config),
        opt_in_flags=opt_in,
    )


__all__ = [
    "apply_cli_overrides",
    "discover_profile_yaml_names",
    "enricher_set_for_profile",
]


_ = Any  # quell ruff/flake8 unused-import suspicion on Any
