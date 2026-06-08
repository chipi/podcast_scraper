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
        "google/long-t5-tglobal-base",
        "google/flan-t5-base",
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


def test_preloaded_pinned_summaries_are_in_the_manifest():
    # The pinned summarizers we actually preload must be in the manifest.
    manifest_ids = {m.model_id for m in mm.REQUIRED_ML_MODELS}
    for model_id in ("google/flan-t5-base", "google/long-t5-tglobal-base"):
        assert cc.get_pinned_revision_for_model(model_id) is not None
        assert model_id in manifest_ids, f"pinned {model_id} not in manifest"


def test_airgapped_thin_summary_is_the_trimmed_manifest_subset():
    # preload_ml_models.py --airgapped-thin reads model_ids_for_tier("airgapped_thin",
    # "summary"); it must be the trimmed bart/led pair and a subset of the test tier.
    air = mm.model_ids_for_tier("airgapped_thin", "summary")
    assert set(air) == {"facebook/bart-base", "allenai/led-base-16384"}
    assert set(air) <= set(mm.model_ids_for_tier("test", "summary"))


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
