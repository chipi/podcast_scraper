"""ML preload manifest (#917): *which* models to preload and *where*.

This is the single source of truth for the **preload policy** — which models the
pipeline downloads and into which tier (dev/test, CI artifact, nightly
production, gated). It deliberately does NOT restate things that already live in
a central place:

- **summary + evidence** model ids are the central ``ModelRegistry`` keys
  (autoresearch-fed via ``scripts/registry/promote_baseline.py`` ->
  ``ModeConfiguration``); their params/capabilities stay in the registry.
- **whisper / spaCy** defaults come from ``config_constants``.
- **pinned revisions** come from ``config_constants.get_pinned_revision_for_model``.

So this module adds only the one dimension the registry lacks: the preload tier.

Consumers: ``scripts/cache/preload_ml_models.py`` (what to download) and the CI
cache validation (``ci_artifact_model_ids`` — the set baked into the ml-models
artifact and loaded offline by the test jobs). The drift guard in
``tests/unit/podcast_scraper/test_ml_model_manifest.py`` keeps every
summary/evidence id here consistent with ``ModelRegistry`` — which is how MiniLM
silently went missing from CI and broke the offline search tests (#897).
"""

from __future__ import annotations

from typing import NamedTuple

from podcast_scraper import config_constants as cc

# Pyannote diarization pipeline id (mirrors config.py's diarization default /
# pyannote_provider.py). Not a ModelRegistry "capability" model.
DIARIZATION_PIPELINE_ID = "pyannote/speaker-diarization-3.1"

MODEL_KINDS = frozenset({"whisper", "spacy", "summary", "embedding", "qa", "nli", "diarization"})
# Kinds whose ids must be central ``ModelRegistry`` entries.
REGISTRY_BACKED_KINDS = frozenset({"summary", "embedding", "qa", "nli"})

# Tier semantics:
#   test        -- preloaded by default ``make preload-ml-models`` (dev/test)
#   ci_artifact -- baked into the ml-models CI artifact AND loaded offline by the
#                  test jobs; CI cache-validation checks exactly this set
#   production  -- preloaded by ``--production`` (nightly / full bake)
#   gated       -- requires HF_TOKEN at download time (pyannote diarization)
MODEL_TIERS = frozenset({"test", "ci_artifact", "production", "gated"})


class MLModelSpec(NamedTuple):
    """A preloadable model: ``model_id`` (HF repo id / Whisper name / spaCy
    package), ``kind`` (one of ``MODEL_KINDS``), and ``tiers`` (subset of
    ``MODEL_TIERS``)."""

    model_id: str
    kind: str
    tiers: frozenset


_T = frozenset({"test", "ci_artifact", "production"})  # core: everywhere
_CI = frozenset({"ci_artifact", "production"})  # artifact + nightly
_PROD = frozenset({"production"})  # nightly / full bake only

REQUIRED_ML_MODELS: tuple[MLModelSpec, ...] = (
    # Whisper (ids from config_constants whisper defaults)
    MLModelSpec(cc.TEST_DEFAULT_WHISPER_MODEL, "whisper", _T),  # tiny.en
    MLModelSpec(cc.PROD_DEFAULT_WHISPER_MODEL, "whisper", _CI),  # base.en
    # spaCy (NER / speaker detection)
    MLModelSpec(cc.TEST_DEFAULT_NER_MODEL, "spacy", _T),  # en_core_web_sm
    # Summarization -- ids are central ModelRegistry keys (params live there)
    MLModelSpec("facebook/bart-base", "summary", _T),
    MLModelSpec("allenai/led-base-16384", "summary", _T),
    MLModelSpec("google/long-t5-tglobal-base", "summary", _CI),
    MLModelSpec("google/flan-t5-base", "summary", _CI),
    MLModelSpec("facebook/bart-large-cnn", "summary", _PROD),
    MLModelSpec("allenai/led-large-16384", "summary", _PROD),
    MLModelSpec("google/long-t5-tglobal-large", "summary", _PROD),
    MLModelSpec("google/flan-t5-large", "summary", _PROD),
    # Evidence stack -- ids from config_constants DEFAULT_* (also registry keys).
    # MiniLM is corpus-wide core -> ci_artifact (the model missing from the #897 CI).
    MLModelSpec(cc.DEFAULT_EMBEDDING_MODEL, "embedding", _T),
    MLModelSpec(cc.DEFAULT_EXTRACTIVE_QA_MODEL, "qa", _PROD),
    MLModelSpec(cc.DEFAULT_NLI_MODEL, "nli", _PROD),
    # Diarization (gated; needs HF_TOKEN)
    MLModelSpec(DIARIZATION_PIPELINE_ID, "diarization", frozenset({"gated", "production"})),
)


def models_for_tier(tier: str) -> tuple[MLModelSpec, ...]:
    """All manifest entries whose ``tiers`` include ``tier``."""
    return tuple(m for m in REQUIRED_ML_MODELS if tier in m.tiers)


def model_ids_for_tier(tier: str, kind: str | None = None) -> list[str]:
    """Model ids in ``tier`` (optionally filtered to one ``kind``)."""
    return [m.model_id for m in models_for_tier(tier) if kind is None or m.kind == kind]


def ci_artifact_model_ids(kind: str | None = None) -> list[str]:
    """Model ids baked into the CI ml-models artifact (optionally one ``kind``).

    The CI cache-validation reads this so the required-model list lives here, not
    in duplicated bash arrays.
    """
    return model_ids_for_tier("ci_artifact", kind=kind)
