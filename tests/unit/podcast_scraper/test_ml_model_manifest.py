"""Drift guard for the ML preload manifest (#917).

Keeps ``model_manifest.REQUIRED_ML_MODELS`` consistent with the other sources of
truth so a new model added in one place can't silently go missing from preload /
CI (the failure mode behind the #897 MiniLM saga):

- summary + evidence ids must be central ``ModelRegistry`` entries (params live
  there, autoresearch-fed) -- the manifest only adds the preload tier;
- the ``DEFAULT_*_MODEL`` constants and summarizer aliases must resolve into it;
- summary models must be in ``ALLOWED_HUGGINGFACE_MODELS``;
- MiniLM must be in the CI artifact tier.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config_constants as cc
from podcast_scraper.providers.ml import model_manifest as mm, summarizer
from podcast_scraper.providers.ml.model_registry import ModelRegistry

pytestmark = pytest.mark.unit


def _spec(model_id: str) -> mm.MLModelSpec | None:
    return next((m for m in mm.REQUIRED_ML_MODELS if m.model_id == model_id), None)


def test_manifest_entries_are_structurally_valid():
    seen: set[str] = set()
    for m in mm.REQUIRED_ML_MODELS:
        assert m.kind in mm.MODEL_KINDS, f"{m.model_id}: unknown kind {m.kind!r}"
        assert m.tiers, f"{m.model_id}: no tiers"
        assert m.tiers <= mm.MODEL_TIERS, f"{m.model_id}: bad tiers {m.tiers - mm.MODEL_TIERS}"
        assert m.model_id not in seen, f"duplicate manifest entry: {m.model_id}"
        seen.add(m.model_id)


def test_summary_and_evidence_ids_are_central_registry_entries():
    # The "one central place" guard: every registry-backed kind must reference a
    # real ModelRegistry id, so the manifest never drifts from the autoresearch
    # registry (it adds only the preload tier, never new model params).
    for m in mm.REQUIRED_ML_MODELS:
        if m.kind in mm.REGISTRY_BACKED_KINDS:
            assert (
                m.model_id in ModelRegistry._registry
            ), f"{m.model_id} ({m.kind}) is not a ModelRegistry entry"


def test_every_default_evidence_model_is_in_the_manifest():
    assert (
        _spec(cc.DEFAULT_EMBEDDING_MODEL) and _spec(cc.DEFAULT_EMBEDDING_MODEL).kind == "embedding"
    )
    assert (
        _spec(cc.DEFAULT_EXTRACTIVE_QA_MODEL) and _spec(cc.DEFAULT_EXTRACTIVE_QA_MODEL).kind == "qa"
    )
    assert _spec(cc.DEFAULT_NLI_MODEL) and _spec(cc.DEFAULT_NLI_MODEL).kind == "nli"


def test_minilm_is_in_the_ci_artifact_tier():
    # Regression guard for the #897 saga: the embedding model the offline search
    # tests load MUST be baked into the CI artifact.
    spec = _spec(cc.DEFAULT_EMBEDDING_MODEL)
    assert spec is not None and "ci_artifact" in spec.tiers
    assert cc.DEFAULT_EMBEDDING_MODEL in mm.ci_artifact_model_ids()


def test_summary_models_are_in_the_security_allowlist():
    for m in mm.REQUIRED_ML_MODELS:
        if m.kind == "summary":
            assert (
                m.model_id in cc.ALLOWED_HUGGINGFACE_MODELS
            ), f"{m.model_id} preloaded but not in ALLOWED_HUGGINGFACE_MODELS"


def test_test_default_summary_aliases_resolve_into_the_manifest():
    manifest_ids = {m.model_id for m in mm.REQUIRED_ML_MODELS}
    for alias in (cc.TEST_DEFAULT_SUMMARY_MODEL, cc.TEST_DEFAULT_SUMMARY_REDUCE_MODEL):
        resolved = summarizer.resolve_model_name(alias)
        assert resolved in manifest_ids, f"{alias} -> {resolved} not in manifest"


def test_ci_artifact_set_covers_the_offline_test_dependencies():
    ci = set(mm.ci_artifact_model_ids())
    expected = {
        cc.TEST_DEFAULT_WHISPER_MODEL,
        cc.PROD_DEFAULT_WHISPER_MODEL,
        cc.TEST_DEFAULT_NER_MODEL,
        "facebook/bart-base",
        "allenai/led-base-16384",
        cc.DEFAULT_EMBEDDING_MODEL,
    }
    missing = expected - ci
    assert not missing, f"CI artifact tier missing required models: {missing}"


def test_evidence_aliases_resolve_to_manifest_ids():
    for alias, kind in (("minilm-l6", "embedding"), ("roberta-squad2", "qa")):
        resolved = ModelRegistry.resolve_evidence_model_id(alias)
        spec = _spec(resolved)
        assert spec is not None and spec.kind == kind, f"alias {alias} -> {resolved}"


def test_pinned_models_are_registry_known():
    # The pin map covers both preloaded (base) and non-preloaded (large) summary
    # variants for reproducibility; all must be ModelRegistry entries even though
    # only the base ones are preloaded (in the manifest).
    for model_id in (
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/long-t5-tglobal-base",
        "google/long-t5-tglobal-large",
    ):
        assert cc.get_pinned_revision_for_model(model_id) is not None
        assert model_id in ModelRegistry._registry, f"pinned {model_id} not in registry"


def test_the_hybrid_summarisers_are_pinned_but_deliberately_not_preloaded():
    """ADR-154 retired the hybrid MAP-REDUCE summariser on 2026-09-24.

    This assertion INVERTED on that date, so it is written out rather than deleted. It used
    to read "pinned therefore preloaded", which was right while hybrid shipped: its MAP
    (long-t5-tglobal-base) and REDUCE (flan-t5-base, 990 MB) models had to be in CI because
    tests/e2e/test_hybrid_ml_provider_e2e.py loaded them for real.

    That spec is gone with the provider, and nothing else in the suite ever asked for either
    model -- all twelve require_transformers_model_cached calls for them lived in that one
    file. Both stay PINNED and registry-known, because they are still supported; they are
    simply not baked into the CI artifact any more. CI carries what the tests exercise.

    Deleting the old assertion outright would have left nothing watching, and something
    re-adding them to REQUIRED_ML_MODELS would put 1 GB back into every run silently.
    """
    manifest_ids = {m.model_id for m in mm.REQUIRED_ML_MODELS}
    for model_id in ("google/flan-t5-base", "google/long-t5-tglobal-base"):
        assert cc.get_pinned_revision_for_model(model_id) is not None, (
            f"{model_id} must stay pinned -- it is supported, just not preloaded"
        )
        assert model_id not in manifest_ids, (
            f"{model_id} is back in the CI preload; ADR-154 retired the only thing using it"
        )


def test_every_pinned_manifest_model_resolves_the_revision_the_loader_will_open():
    """The preload and the runtime must ask for the SAME revision.

    When they disagree, the preload builds a cache at one revision and the loader opens
    another -- the cache looks complete and is unreadable. That is precisely how six e2e
    specs failed for two days while every upstream signal reported success.
    """
    for spec in mm.REQUIRED_ML_MODELS:
        if spec.kind in ("whisper", "spacy"):
            continue
        rev = cc.get_pinned_revision_for_model(spec.model_id)
        if rev is None:
            continue
        assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), (
            f"{spec.model_id} has a pin that is not a full commit sha: {rev!r}"
        )


def test_airgapped_thin_summary_is_the_trimmed_pair():
    """``preload_ml_models.py --airgapped-thin`` preloads exactly this pair.

    This used to also assert ``set(air) <= set(model_ids_for_tier("test", "summary"))``, which
    stopped holding when LED left the test tier. The replacement written first was
    ``set(air) <= {m.model_id for m in REQUIRED_ML_MODELS}`` — which CANNOT FAIL, because
    ``models_for_tier`` is a filter OVER ``REQUIRED_ML_MODELS``. A tautology in the place a
    dropped guard used to be is worse than no guard: it reads as coverage.

    What is worth pinning here is the pair itself. The relationship to the test tier moved to
    ``test_the_test_tier_is_the_loadable_subset_of_airgapped_thin``, where it is asserted in the
    direction that now runs.
    """
    air = mm.model_ids_for_tier("airgapped_thin", "summary")
    assert set(air) == {"facebook/bart-base", "allenai/led-base-16384"}
    for model_id in air:
        assert cc.get_pinned_revision_for_model(model_id), f"{model_id} preloaded unpinned"


def test_the_test_tier_holds_only_loadable_checkpoints():
    """Nothing in the ``test`` tier needs torch >= 2.6 to load.

    The ``test`` tier is what a developer machine preloads, so every entry has to be loadable on
    whatever torch that machine can install. ``transformers >= 4.56`` refuses ``torch.load``
    below torch 2.6 (PYSEC-2025-41), and the checkpoints in ``PICKLE_ONLY_CHECKPOINTS`` have no
    safetensors build, so they cannot be loaded any other way.

    This is a property of the TIER, stated on its own terms — not a relationship to some other
    tier that happens to share models with it. A pickle-only entry here makes
    ``make preload-ml-models`` a hard failure rather than a slow one, and takes ``make ci`` with
    it.
    """
    for kind in ("summary", "embedding", "qa", "nli"):
        for model_id in mm.model_ids_for_tier("test", kind):
            assert model_id not in mm.PICKLE_ONLY_CHECKPOINTS, (
                f"{model_id} is pickle-only and cannot load below torch 2.6, so it must not be "
                "in the `test` tier — move it to ci_artifact/production, which run on hosts "
                "where torch is current"
            )


def test_preload_evidence_defaults_are_manifest_ids():
    # Reverse drift guard: every DEFAULT_*_MODEL the preload script downloads for the
    # evidence stack must be in the manifest, so a preloaded model can't silently
    # fall out of the single source of truth.
    manifest_ids = {m.model_id for m in mm.REQUIRED_ML_MODELS}
    for model_id in (
        cc.DEFAULT_EMBEDDING_MODEL,
        cc.DEFAULT_EXTRACTIVE_QA_MODEL,
        cc.DEFAULT_NLI_MODEL,
    ):
        assert model_id in manifest_ids, f"preloaded evidence model {model_id} not in manifest"


def test_production_evidence_tier_matches_preload_constants():
    # Tighter contract than membership above. preload_ml_models.py --production fetches
    # the evidence stack via cc.DEFAULT_*_MODEL, while verify_required_models.py checks
    # the cache against model_ids_for_tier("production", <kind>). If a production-tier
    # evidence id is swapped in the manifest WITHOUT updating the constant (or vice
    # versa), preload downloads one model and the verifier checks a different one ->
    # CI cache validation fails for everyone. Pin the two together per kind.
    for kind, constant in (
        ("embedding", cc.DEFAULT_EMBEDDING_MODEL),
        ("qa", cc.DEFAULT_EXTRACTIVE_QA_MODEL),
        ("nli", cc.DEFAULT_NLI_MODEL),
    ):
        assert mm.model_ids_for_tier("production", kind) == [constant], (
            f"production-tier {kind} manifest id diverged from the preload constant "
            f"({constant!r}); preload and the cache verifier would disagree"
        )
