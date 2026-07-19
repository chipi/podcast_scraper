"""Unit tests for the CIL MCP tools (RFC-095 slice 3) — cil_queries mocked."""

from __future__ import annotations

import pytest

from podcast_scraper.mcp.context import CorpusContext
from podcast_scraper.mcp.tools import cil

pytestmark = pytest.mark.unit


def test_person_profile_wraps_with_subject(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    captured = {}

    def fake(root, anchor, person):
        captured["args"] = (root, anchor, person)
        return {"insights": [{"id": "i1"}]}

    monkeypatch.setattr("podcast_scraper.server.cil_queries.person_profile", fake)
    out = cil.person_profile(ctx, "person:jane")
    assert out["subject"] == "person:jane"
    assert out["profile"] == {"insights": [{"id": "i1"}]}
    # root == anchor == corpus dir.
    assert captured["args"] == (str(tmp_path.resolve()), str(tmp_path.resolve()), "person:jane")


def test_topic_timeline_wraps_list(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    monkeypatch.setattr(
        "podcast_scraper.server.cil_queries.topic_timeline",
        lambda root, anchor, topic: [{"episode": "e1"}, {"episode": "e2"}],
    )
    out = cil.topic_timeline(ctx, "topic:ai")
    assert out["subject"] == "topic:ai"
    assert len(out["timeline"]) == 2


def test_position_arc_wraps_both_subjects(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    monkeypatch.setattr(
        "podcast_scraper.server.cil_queries.position_arc",
        lambda root, anchor, person, topic, **kw: [{"stance": "for"}],
    )
    out = cil.position_arc(ctx, "person:jane", "topic:ai")
    assert out["subject_person"] == "person:jane"
    assert out["subject_topic"] == "topic:ai"
    assert out["arc"] == [{"stance": "for"}]


def test_person_profile_rejects_placeholder(tmp_path, monkeypatch) -> None:
    # #1193: a person:speaker-NN placeholder gets no profile — and the underlying query is not even
    # called (the guard short-circuits at the MCP boundary).
    ctx = CorpusContext.from_path(tmp_path)

    def boom(*a, **k):
        raise AssertionError("cil_queries must not be called for a placeholder")

    monkeypatch.setattr("podcast_scraper.server.cil_queries.person_profile", boom)
    out = cil.person_profile(ctx, "person:speaker-02")
    assert out == {"subject": "person:speaker-02", "profile": {}}


def test_position_arc_rejects_placeholder(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)

    def boom(*a, **k):
        raise AssertionError("cil_queries must not be called for a placeholder")

    monkeypatch.setattr("podcast_scraper.server.cil_queries.position_arc", boom)
    out = cil.position_arc(ctx, "person:speaker-02", "topic:ai")
    assert out == {"subject_person": "person:speaker-02", "subject_topic": "topic:ai", "arc": {}}
