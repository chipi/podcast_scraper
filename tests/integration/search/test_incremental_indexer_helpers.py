"""Unit coverage for the D8/D2 incremental-indexer helpers (fingerprint + scope key + persistence).

Pure-function branches of ``two_tier_indexer`` that the integration build tests don't exercise:
fingerprint determinism/sensitivity, scope-key resolution + fallback + None, fingerprint file
load (missing/malformed), and the D2 ``os.utime`` failure being non-fatal.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

pytest.importorskip("lancedb")

from podcast_scraper.search import two_tier_indexer as tti  # noqa: E402


def _rows(ep_id="ep1", feed="s", text="hello"):
    return [
        (f"insight:{ep_id}", text, {"doc_type": "insight", "episode_id": ep_id, "feed_id": feed})
    ]


def test_fingerprint_is_deterministic_and_order_independent():
    a = tti._episode_fingerprint(_rows(), "m")
    b = tti._episode_fingerprint(list(reversed(_rows() + _rows(ep_id="ep2"))), "m")
    c = tti._episode_fingerprint(_rows() + _rows(ep_id="ep2"), "m")
    assert a == tti._episode_fingerprint(_rows(), "m")  # stable
    assert b == c  # order-independent


def test_fingerprint_changes_with_content_and_model():
    base = tti._episode_fingerprint(_rows(text="x"), "m")
    assert base != tti._episode_fingerprint(_rows(text="y"), "m")  # content
    assert base != tti._episode_fingerprint(_rows(text="x"), "other-model")  # model


def test_scope_key_from_rows_then_doc_fallback_then_none():
    # resolves (feed_id, episode_id) from the rows' meta
    assert tti._episode_scope_key(_rows("epR", "f"), {}) == tti.index_fingerprint_scope_key(
        "f", "epR"
    )
    # rows carry no episode_id → fall back to the loaded metadata's episode
    no_id_rows = [("q:1", "t", {"doc_type": "quote"})]
    assert tti._episode_scope_key(
        no_id_rows, {"episode": {"episode_id": "epD", "feed_id": "f"}}
    ) == tti.index_fingerprint_scope_key("f", "epD")
    # neither rows nor doc resolve an episode_id → None (never skip)
    assert tti._episode_scope_key(no_id_rows, {"episode": {}}) is None


def test_load_fingerprints_missing_or_malformed_returns_empty(tmp_path):
    lance = tmp_path / "search" / "lance_index"
    lance.mkdir(parents=True)
    assert tti._load_episode_fingerprints(lance) == {}  # absent
    tti._fingerprints_path(lance).write_text("not json", encoding="utf-8")
    assert tti._load_episode_fingerprints(lance) == {}  # malformed
    tti._write_episode_fingerprints(lance, {"k": "v"})
    assert tti._load_episode_fingerprints(lance) == {"k": "v"}  # round-trip


def test_utime_failure_is_non_fatal(tmp_path, monkeypatch, caplog):
    """D2: bumping the index dir mtime must never fail a build."""

    def _boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(tti.os, "utime", _boom)
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    meta = corpus / "metadata" / "ep1.metadata.json"
    monkeypatch.setattr(tti, "discover_metadata_files", lambda root: [meta])
    monkeypatch.setattr(tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": "ep1"}})
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: corpus)
    monkeypatch.setattr(
        tti,
        "_collect_docs_for_episode",
        lambda *a, **k: [
            ("insight:ep1", "t", {"doc_type": "insight", "episode_id": "ep1", "feed_id": "s"})
        ],
    )
    monkeypatch.setattr(tti, "_embed", lambda text, m, *, allow_download: [0.1] * 8)

    stats = tti.build_two_tier_index(corpus, corpus / "search" / "lance_index", drop_existing=True)
    assert stats.episodes == 1  # build succeeded despite os.utime raising
