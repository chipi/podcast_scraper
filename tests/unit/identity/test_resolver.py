"""Unit tests for the file-local entity resolver (identity/resolver.py, #849)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from podcast_scraper.identity.resolver import (
    build_entity_registry,
    clear_entity_resolver_cache,
    EntityRegistry,
    EntityResolver,
    get_entity_resolver,
    ResolveResult,
)

pytestmark = pytest.mark.unit


class FakeEmbedder:
    """Deterministic embedder so exact + fuzzy paths are reproducible offline.

    Maps a few known strings near the "altman" axis (so a misspelling resolves
    fuzzily) and everything else to an orthogonal axis (cosine 0 → no match).
    """

    # 5-dim so each known registry name gets its own axis and unknown text lands
    # on a reserved axis (e3) orthogonal to every known name → cosine 0 → no match.
    _MAP = {
        "samuel altman": [1.0, 0.0, 0.0, 0.0, 0.0],
        "sam altman": [0.99, 0.0, 0.0, 0.0, 0.14],
        "samuel altmann": [0.97, 0.0, 0.0, 0.0, 0.24],  # misspelling → still near e0
        "openai": [0.0, 1.0, 0.0, 0.0, 0.0],
        "ai regulation": [0.0, 0.0, 1.0, 0.0, 0.0],
    }
    _UNKNOWN = [0.0, 0.0, 0.0, 1.0, 0.0]

    def _vec(self, text: str) -> list[float]:
        return self._MAP.get(" ".join(str(text).lower().split()), list(self._UNKNOWN))

    def encode(self, texts, normalize_embeddings=True):  # noqa: D401 - mimic ST signature
        items = [texts] if isinstance(texts, str) else list(texts)
        arr = np.asarray([self._vec(t) for t in items], dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr


def _identity(cid, type_, display, aliases=None):
    return {
        "id": cid,
        "type": type_,
        "display_name": display,
        "aliases": list(aliases or []),
        "sources": {"gi": True, "kg": False},
    }


def _registry(**kwargs) -> EntityRegistry:
    identities = {
        "person:samuel-altman": _identity(
            "person:samuel-altman", "person", "Samuel Altman", ["Sam Altman", "sama"]
        ),
        "org:openai": _identity("org:openai", "org", "OpenAI"),
        "topic:ai-regulation": _identity("topic:ai-regulation", "topic", "AI regulation"),
    }
    freq = Counter({"person:samuel-altman": 5, "org:openai": 3, "topic:ai-regulation": 2})
    return EntityRegistry.from_identities(identities, freq, embedder=FakeEmbedder(), **kwargs)


def _resolver(**kwargs) -> EntityResolver:
    return EntityResolver(_registry(**kwargs))


# --- exact tiers ---------------------------------------------------------------


def test_resolve_exact_canonical_id():
    r = _resolver()
    assert r.resolve("person:samuel-altman") == "person:samuel-altman"
    assert r.resolve_detail("person:samuel-altman").method == "exact_id"


def test_resolve_slug_match():
    r = _resolver()
    # slugify("OpenAI") == "openai" -> org:openai
    res = r.resolve_detail("OpenAI")
    assert res.id == "org:openai"
    assert res.method == "slug"
    assert r.resolve("AI regulation") == "topic:ai-regulation"


def test_resolve_alias_match():
    r = _resolver()
    res = r.resolve_detail("Sam Altman")  # alias of person:samuel-altman
    assert res.id == "person:samuel-altman"
    assert res.method == "alias"


# --- fuzzy + conservative ------------------------------------------------------


def test_resolve_fuzzy_hit():
    r = _resolver()
    res = r.resolve_detail("Samuel Altmann")  # misspelling, not an exact name/slug
    assert res is not None
    assert res.id == "person:samuel-altman"
    assert res.method == "fuzzy"
    assert res.score >= 0.75


def test_resolve_fuzzy_miss_returns_none():
    r = _resolver()
    assert r.resolve("zxcvb qwerty nonsense") is None


def test_resolve_empty_and_blank_return_none():
    r = _resolver()
    assert r.resolve("") is None
    assert r.resolve("   ") is None
    assert r.resolve_detail("") is None


def test_resolve_below_threshold_is_conservative():
    # Raise threshold above the misspelling similarity → no fuzzy match.
    r = _resolver(fuzzy_threshold=0.999)
    assert r.resolve("Samuel Altmann") is None


# --- precedence + tie-break ----------------------------------------------------


def test_slug_precedence_person_over_topic():
    identities = {
        "person:dup-term": _identity("person:dup-term", "person", "Dup Person"),
        "topic:dup-term": _identity("topic:dup-term", "topic", "Dup Topic"),
    }
    reg = EntityRegistry.from_identities(
        identities, Counter({"person:dup-term": 1, "topic:dup-term": 9}), embedder=FakeEmbedder()
    )
    res = EntityResolver(reg).resolve_detail("Dup Term")  # slug "dup-term"
    assert res.id == "person:dup-term"  # person wins regardless of topic freq
    assert res.method == "slug"


def test_alias_collision_prefers_higher_frequency():
    identities = {
        "person:acme-a": _identity("person:acme-a", "person", "AcmeA", ["Acme"]),
        "person:acme-b": _identity("person:acme-b", "person", "AcmeB", ["Acme"]),
    }
    reg = EntityRegistry.from_identities(
        identities, Counter({"person:acme-a": 2, "person:acme-b": 5}), embedder=FakeEmbedder()
    )
    assert EntityResolver(reg).resolve("Acme") == "person:acme-b"


# --- registry build from real artifacts ----------------------------------------


def test_build_registry_from_fixture_corpus():
    corpus = Path("tests/fixtures/gil_kg_ci_enforce")
    reg = build_entity_registry(corpus, embedder=FakeEmbedder())
    assert len(reg) >= 1
    # The fixture gi.json carries Topic node topic:ci-policy (label "Climate policy").
    assert "topic:ci-policy" in reg.records
    assert EntityResolver(reg).resolve("Climate policy") == "topic:ci-policy"


def test_resolve_detail_returns_resolveresult_type():
    r = _resolver()
    assert isinstance(r.resolve_detail("person:samuel-altman"), ResolveResult)
    assert r.resolve_detail("definitely not present") is None


def test_get_entity_resolver_caches_registry():
    corpus = "tests/fixtures/gil_kg_ci_enforce"
    clear_entity_resolver_cache()
    r1 = get_entity_resolver(corpus, embedder=FakeEmbedder())
    r2 = get_entity_resolver(corpus, embedder=FakeEmbedder())
    assert r1.registry is r2.registry  # registry built once, reused
    clear_entity_resolver_cache()
    r3 = get_entity_resolver(corpus, embedder=FakeEmbedder())
    assert r3.registry is not r1.registry


def _corpus_copy(tmp_path):
    import shutil

    src = Path("tests/fixtures/gil_kg_ci_enforce")
    dst = tmp_path / "corpus"
    shutil.copytree(src, dst)
    return dst


def _bump_corpus(corpus: Path, when: float) -> None:
    import os

    stamp = corpus / "corpus_run_summary.json"
    stamp.write_text("{}", encoding="utf-8")
    os.utime(stamp, (when, when))


def test_a_changed_corpus_refreshes_the_registry_in_the_background(tmp_path):
    """Prod 2026-10-13: the registry was built once per process and never refreshed."""
    import podcast_scraper.identity.resolver as res

    corpus = _corpus_copy(tmp_path)
    _bump_corpus(corpus, 1_000_000.0)
    clear_entity_resolver_cache()
    r1 = get_entity_resolver(corpus, embedder=FakeEmbedder())

    _bump_corpus(corpus, 2_000_000.0)  # an ingest / migration rewrote the corpus
    r2 = get_entity_resolver(corpus, embedder=FakeEmbedder())
    assert r2.registry is r1.registry  # the stale one answers at once, no in-request rebuild
    assert res._last_refresh is not None
    res._last_refresh.join(timeout=30)

    r3 = get_entity_resolver(corpus, embedder=FakeEmbedder())
    assert r3.registry is not r1.registry  # the background rebuild replaced it
    r4 = get_entity_resolver(corpus, embedder=FakeEmbedder())
    assert r4.registry is r3.registry  # and it is fresh now: no further rebuild


def test_an_unchanged_corpus_starts_no_rebuild(tmp_path):
    import podcast_scraper.identity.resolver as res

    corpus = _corpus_copy(tmp_path)
    _bump_corpus(corpus, 1_000_000.0)
    clear_entity_resolver_cache()
    get_entity_resolver(corpus, embedder=FakeEmbedder())
    res._last_refresh = None
    get_entity_resolver(corpus, embedder=FakeEmbedder())
    assert res._last_refresh is None


def test_warm_up_builds_the_registry_before_the_first_lookup(tmp_path, monkeypatch):
    import podcast_scraper.identity.resolver as res

    corpus = _corpus_copy(tmp_path)
    clear_entity_resolver_cache()
    res.warm_entity_resolver(corpus, embedder=FakeEmbedder()).join(timeout=30)
    monkeypatch.setattr(
        res, "build_entity_registry", lambda *a, **k: pytest.fail("rebuilt on first lookup")
    )
    assert get_entity_resolver(corpus, embedder=FakeEmbedder()).registry.records
