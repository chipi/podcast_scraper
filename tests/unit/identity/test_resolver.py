"""Unit tests for the file-local entity resolver (identity/resolver.py, #849)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from podcast_scraper.identity.resolver import (
    build_entity_registry,
    EntityRegistry,
    EntityResolver,
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
