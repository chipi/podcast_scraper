"""ML preload manifest (#917): *which* models to preload and *where*.

This is the single source of truth for the **preload policy** — which models the
pipeline downloads and into which tier (dev/test, CI artifact, nightly
production, gated). It deliberately does NOT restate things that already live in
a central place:

- **summary + evidence** model ids are the central ``ModelRegistry`` keys
  (autoresearch-fed via ``eval-data/scripts/registry/promote_baseline.py`` ->
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
# pyannote_provider.py). A ModelRegistry _DIARIZATION_OPTIONS stage, but preloaded
# here (not a "capability" model). community-1 (v4) is the promoted default.
DIARIZATION_PIPELINE_ID = "pyannote/speaker-diarization-community-1"

MODEL_KINDS = frozenset({"whisper", "spacy", "summary", "embedding", "qa", "nli", "diarization"})
# Kinds whose ids must be central ``ModelRegistry`` entries.
REGISTRY_BACKED_KINDS = frozenset({"summary", "embedding", "qa", "nli"})

# Tier semantics:
#   test           -- preloaded by default ``make preload-ml-models`` (dev/test)
#   ci_artifact    -- baked into the ml-models CI artifact AND loaded offline by the
#                     test jobs; CI cache-validation checks exactly this set
#   production     -- preloaded by ``--production`` (nightly / full bake)
#   airgapped_thin -- the trimmed summarizer subset preloaded by ``--airgapped-thin``
#                     (matches config/profiles/airgapped_thin.yaml). Whisper/spaCy/
#                     evidence for that bundle come from the shared test-tier defaults,
#                     so only the summary subset is tagged here.
#   gated          -- requires HF_TOKEN at download time (pyannote diarization)
MODEL_TIERS = frozenset({"test", "ci_artifact", "production", "airgapped_thin", "gated"})


class MLModelSpec(NamedTuple):
    """A preloadable model: ``model_id`` (HF repo id / Whisper name / spaCy
    package), ``kind`` (one of ``MODEL_KINDS``), and ``tiers`` (subset of
    ``MODEL_TIERS``)."""

    model_id: str
    kind: str
    tiers: frozenset


_T = frozenset({"test", "ci_artifact", "production"})  # core: everywhere
_T_AIR = _T | {"airgapped_thin"}  # core + the trimmed airgapped-thin summarizers
_CI = frozenset({"ci_artifact", "production"})  # artifact + nightly
_CI_AIR = _CI | {"airgapped_thin"}  # artifact + nightly + airgapped-thin, but NOT local test
_PROD = frozenset({"production"})  # nightly / full bake only

# WHY TWO OF THE SUMMARISERS ARE NOT IN THE ``test`` TIER (2026-09-24)
#
# ``allenai/led-base-16384`` and ``google/long-t5-tglobal-base`` ship ONLY
# ``pytorch_model.bin`` — no safetensors — so loading either one unpickles through
# ``torch.load``. ``transformers >= 4.56`` refuses that below torch 2.6, citing
# PYSEC-2025-41 / CVE-2025-32434, and it is right to: ``weights_only=True`` does not
# close that hole.
#
# On x86_64 macOS the newest torch wheel that exists is 2.2.2, so ``make
# preload-ml-models`` — and therefore ``make ci`` — could not complete on that host at
# all. Not a skipped stage: a hard failure on a cached model.
#
# The ``test`` tier is what a DEVELOPER MACHINE preloads. It is now safetensors-only, so
# it loads anywhere. CI, nightly and airgapped-thin keep both models: they run on Linux
# where torch is current and the guard never fires. Nothing about the airgapped profile
# changes — it still needs a long-context local REDUCE and LED is still the only one.
#
# Removing them from ``test`` does not make them unpinned or unreachable: both carry a
# SHA (ADR-155) and both stay in ``REQUIRED_ML_MODELS``.

#: Checkpoints that publish ONLY ``pytorch_model.bin`` — no ``model.safetensors`` at the
#: revision we pin. Loading one unpickles through ``torch.load``, which ``transformers >= 4.56``
#: refuses below torch 2.6 (PYSEC-2025-41 / CVE-2025-32434). Stated here because it is a fact
#: about the checkpoints, not about any one machine: the same refusal fires anywhere torch is
#: older than 2.6, and x86_64 macOS is simply where it fires today (newest wheel: 2.2.2).
#:
#: Checked against the Hub on 2026-09-24. No safetensors build of these WEIGHTS exists — the
#: community copies are pickle too, and near-name repos like ``led-base-16384-ms2`` are
#: fine-tunes. So this is a durable property, not a "pending upstream" note.
#:
#: The ``test`` tier must not contain any of these: it is what a developer machine preloads, and
#: a model it cannot load turns ``make preload-ml-models`` — and therefore ``make ci`` — into a
#: hard failure. ``test_the_test_tier_holds_only_loadable_checkpoints`` enforces that.
PICKLE_ONLY_CHECKPOINTS: frozenset[str] = frozenset(
    {
        "google/pegasus-large",
        "google/pegasus-cnn_dailymail",
        "google/pegasus-xsum",
        "google/long-t5-tglobal-base",
        "google/long-t5-tglobal-large",
        "allenai/led-base-16384",
        "allenai/led-large-16384",
        "sshleifer/distilbart-cnn-12-6",
    }
)

REQUIRED_ML_MODELS: tuple[MLModelSpec, ...] = (
    # Whisper (ids from config_constants whisper defaults)
    MLModelSpec(cc.TEST_DEFAULT_WHISPER_MODEL, "whisper", _T),  # tiny.en
    MLModelSpec(cc.PROD_DEFAULT_WHISPER_MODEL, "whisper", _CI),  # base.en
    # spaCy (NER / speaker detection)
    MLModelSpec(cc.TEST_DEFAULT_NER_MODEL, "spacy", _T),  # en_core_web_sm
    # _trf is the prod NER default (PROD_DEFAULT_NER_MODEL); preloaded into
    # CI artifact + production-tier bakes per #984 (+13 pp v2 spec recall vs
    # _sm at ~2x latency, still sub-second). _sm stays the dev/test default
    # to keep the cycle quick and the install footprint small.
    MLModelSpec(cc.PROD_DEFAULT_NER_MODEL, "spacy", _CI),  # en_core_web_trf
    # Summarization -- ids are central ModelRegistry keys (params live there).
    # These four are the actual preloaded summarizers (the default dev preload and
    # --production both bake all four); the larger *-large variants are registry/
    # ALLOWED-known and revision-pinned but are NOT preloaded by default, so they
    # are intentionally absent from this preload manifest.
    MLModelSpec("facebook/bart-base", "summary", _T_AIR),  # airgapped-thin bart-small
    # pickle-only -> not in `test`; see the note above the tier constants
    MLModelSpec("allenai/led-base-16384", "summary", _CI_AIR),  # airgapped-thin long-fast
    MLModelSpec("google/long-t5-tglobal-base", "summary", _CI),
    MLModelSpec("google/flan-t5-base", "summary", _T),
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


def torch_refuses_pickle_weights() -> bool:
    """Whether this runtime's transformers will refuse a pickle-only checkpoint.

    ``transformers >= 4.56`` will not call ``torch.load`` below torch 2.6, citing
    PYSEC-2025-41 / CVE-2025-32434. It raises rather than degrading, so on such a runtime a
    checkpoint in :data:`PICKLE_ONLY_CHECKPOINTS` cannot be loaded at all — not slowly, not
    with a warning: not at all.

    Returns False when torch is absent, because then no summariser is running anyway and
    the caller's model choice is moot.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        major, minor = (int(p) for p in str(torch.__version__).split(".")[:2])
    except (TypeError, ValueError):
        return False
    return (major, minor) < (2, 6)


def checkpoint_is_loadable_here(model_id: str) -> bool:
    """Whether ``model_id`` can actually be loaded by this runtime.

    ``PICKLE_ONLY_CHECKPOINTS`` existed as data that nothing consulted: the manifest used it
    to decide what to PRELOAD, and the model selectors went on naming those same checkpoints
    as defaults. A model excluded from one path and still chosen by another is excluded from
    neither — on an x86_64 Mac (torch caps at 2.2.2) that produced a hard ValueError deep in
    ``from_pretrained``, which is a crash where a substitution would do.
    """
    return not (model_id in PICKLE_ONLY_CHECKPOINTS and torch_refuses_pickle_weights())
