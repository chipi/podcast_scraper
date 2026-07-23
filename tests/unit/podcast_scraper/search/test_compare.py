"""Unit tests for search/compare.py — Search v3 §S8.

Compare orchestrator wraps ``build_briefing_pack`` twice, one per picker
slot. The tests stub ``structured_corpus_search`` (retrieval is
already covered by capability tests) and exercise the compare-specific
behaviours: scope mapping, ungrounded-side handling, and the judge
summary mute rule (RFC-107 §S8 acceptance).

No LanceDB touch; no network; no LLM call. ``make lint-search-v3``
covers the module's forbidden-import surface at the repo level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.search.compare import (
    _judge_summary,
    _scope_for_subject,
    BriefingPack,
    compare_subjects,
    SubjectRef,
)


def _hit(
    doc_id: str,
    text: str,
    *,
    source_tier: str = "insight",
    episode_id: str = "ep1",
    show_id: str = "show1",
    publish_date: str = "2026-06-01",
    confidence: float = 0.8,
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "score": 0.9,
        "rank": 0,
        "source_tier": source_tier,
        "signal": "rrf",
        "text": text,
        "metadata": {
            "episode_id": episode_id,
            "show_id": show_id,
            "publish_date": publish_date,
            "confidence": confidence,
        },
    }


def _stub_search(
    monkeypatch: pytest.MonkeyPatch,
    per_call_results: List[List[Dict[str, Any]]],
    *,
    error: str | None = None,
) -> List[Dict[str, Any]]:
    """Install a fake ``structured_corpus_search`` that returns the given
    result lists on successive calls. Records each call's kwargs so tests
    can assert scope mapping."""
    calls: List[Dict[str, Any]] = []
    iterator = iter(per_call_results)

    def _fake(root: Path, query: str, **kwargs: Any) -> Dict[str, Any]:
        calls.append({"root": root, "query": query, **kwargs})
        if error:
            return {"results": [], "error": error, "detail": None, "query_type": "semantic"}
        try:
            page = next(iterator)
        except StopIteration:
            page = []
        return {
            "results": page,
            "error": None,
            "detail": None,
            "query_type": "semantic",
            "lift_stats": {"transcript_hits_returned": 0, "lift_applied": 0},
        }

    monkeypatch.setattr(
        "podcast_scraper.search.capability.structured_corpus_search",
        _fake,
    )
    return calls


# --------------------------------------------------------------------------
# _scope_for_subject
# --------------------------------------------------------------------------


class TestScopeForSubject:
    def test_person_maps_to_speaker(self) -> None:
        assert _scope_for_subject(SubjectRef(kind="person", id="Alice")) == {"speaker": "Alice"}

    def test_topic_maps_to_topic(self) -> None:
        assert _scope_for_subject(SubjectRef(kind="topic", id="topic:compute")) == {
            "topic": "topic:compute"
        }

    def test_episode_maps_to_episode_id(self) -> None:
        assert _scope_for_subject(SubjectRef(kind="episode", id="ep-a")) == {"episode_id": "ep-a"}

    def test_feed_maps_to_feed(self) -> None:
        assert _scope_for_subject(SubjectRef(kind="feed", id="sha256:abc")) == {
            "feed": "sha256:abc"
        }

    def test_show_aliases_to_feed(self) -> None:
        """``show`` is an alias for feed — same underlying scope kwarg."""
        assert _scope_for_subject(SubjectRef(kind="show", id="sha256:abc")) == {
            "feed": "sha256:abc"
        }


# --------------------------------------------------------------------------
# _judge_summary
# --------------------------------------------------------------------------


def _pack(
    *,
    subject: SubjectRef,
    grounded: bool,
    confidence: float = 0.5,
    episode_count: int = 3,
) -> BriefingPack:
    return BriefingPack(
        subject=subject,
        query="q",
        grounded=grounded,
        confidence_p50=confidence,
        coverage_summary={"episode_count": episode_count, "show_ids": [], "date_range": None},
    )


class TestJudgeSummary:
    def test_muted_when_a_ungrounded(self) -> None:
        a = _pack(subject=SubjectRef(kind="person", id="A"), grounded=False)
        b = _pack(subject=SubjectRef(kind="person", id="B"), grounded=True)
        assert _judge_summary(a, b) is None

    def test_muted_when_b_ungrounded(self) -> None:
        a = _pack(subject=SubjectRef(kind="person", id="A"), grounded=True)
        b = _pack(subject=SubjectRef(kind="person", id="B"), grounded=False)
        assert _judge_summary(a, b) is None

    def test_muted_when_both_ungrounded(self) -> None:
        a = _pack(subject=SubjectRef(kind="person", id="A"), grounded=False)
        b = _pack(subject=SubjectRef(kind="person", id="B"), grounded=False)
        assert _judge_summary(a, b) is None

    def test_higher_confidence_side_named_first(self) -> None:
        a = _pack(subject=SubjectRef(kind="person", id="A"), grounded=True, confidence=0.9)
        b = _pack(subject=SubjectRef(kind="person", id="B"), grounded=True, confidence=0.5)
        summary = _judge_summary(a, b) or ""
        assert summary.startswith("A shows higher confidence")
        assert "0.90 vs 0.50" in summary

    def test_uses_label_when_present(self) -> None:
        a = _pack(
            subject=SubjectRef(kind="person", id="person:alice", label="Alice"),
            grounded=True,
            confidence=0.7,
        )
        b = _pack(
            subject=SubjectRef(kind="person", id="person:bob", label="Bob"),
            grounded=True,
            confidence=0.6,
        )
        summary = _judge_summary(a, b) or ""
        assert "Alice" in summary
        assert "Bob" in summary
        assert "person:alice" not in summary

    def test_equal_confidence_reports_tie(self) -> None:
        a = _pack(subject=SubjectRef(kind="person", id="A"), grounded=True, confidence=0.6)
        b = _pack(subject=SubjectRef(kind="person", id="B"), grounded=True, confidence=0.6)
        summary = _judge_summary(a, b) or ""
        assert "equal confidence" in summary

    def test_reports_episode_coverage(self) -> None:
        a = _pack(
            subject=SubjectRef(kind="person", id="A"),
            grounded=True,
            confidence=0.9,
            episode_count=8,
        )
        b = _pack(
            subject=SubjectRef(kind="person", id="B"),
            grounded=True,
            confidence=0.5,
            episode_count=3,
        )
        summary = _judge_summary(a, b) or ""
        assert "episode coverage 8 (A) vs 3 (B)" in summary


# --------------------------------------------------------------------------
# compare_subjects — orchestration
# --------------------------------------------------------------------------


class TestCompareSubjects:
    def test_both_sides_grounded_returns_judge_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_search(
            monkeypatch,
            [
                [_hit("insight:a1", "insight A", confidence=0.9, episode_id="epA")],
                [_hit("insight:b1", "insight B", confidence=0.5, episode_id="epB")],
            ],
        )
        outcome = compare_subjects(
            tmp_path,
            SubjectRef(kind="person", id="person:alice", label="Alice"),
            SubjectRef(kind="person", id="person:bob", label="Bob"),
            q="compute",
        )
        assert outcome.pack_a.grounded is True
        assert outcome.pack_b.grounded is True
        assert outcome.pack_a.top_insight_id == "insight:a1"
        assert outcome.pack_b.top_insight_id == "insight:b1"
        assert outcome.judge_summary is not None
        assert "Alice" in outcome.judge_summary

    def test_ungrounded_side_produces_placeholder_and_mutes_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty result set → grounded=false pack; judge summary muted."""
        _stub_search(
            monkeypatch,
            [
                [_hit("insight:a1", "insight A")],
                [],  # subject B has zero hits — ungrounded side
            ],
        )
        outcome = compare_subjects(
            tmp_path,
            SubjectRef(kind="person", id="Alice"),
            SubjectRef(kind="person", id="Bob"),
            q="topic",
        )
        assert outcome.pack_a.grounded is True
        assert outcome.pack_b.grounded is False
        assert outcome.pack_b.top_insight_id is None
        assert outcome.pack_b.result_count == 0
        assert outcome.judge_summary is None

    def test_search_error_produces_ungrounded_pack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``no_index`` (or any other) error degrades to an empty pack."""
        _stub_search(monkeypatch, [[]], error="no_index")
        outcome = compare_subjects(
            tmp_path,
            SubjectRef(kind="person", id="A"),
            SubjectRef(kind="person", id="B"),
            q="x",
        )
        assert outcome.pack_a.grounded is False
        assert outcome.pack_b.grounded is False
        assert outcome.judge_summary is None

    def test_scope_kwargs_reach_search_layer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Person → speaker=, topic → topic=; each side's scope is
        threaded into ``structured_corpus_search`` verbatim."""
        calls = _stub_search(
            monkeypatch,
            [
                [_hit("insight:a1", "A")],
                [_hit("insight:b1", "B")],
            ],
        )
        compare_subjects(
            tmp_path,
            SubjectRef(kind="person", id="Alice"),
            SubjectRef(kind="topic", id="topic:policy"),
            q="",
            top_k=15,
        )
        assert calls[0]["speaker"] == "Alice"
        assert calls[0]["top_k"] == 15
        assert "topic" not in calls[0]
        assert calls[1]["topic"] == "topic:policy"
        assert calls[1]["top_k"] == 15
        assert "speaker" not in calls[1]

    def test_empty_shared_q_falls_back_to_subject_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No shared query → each side searches for its own subject label."""
        calls = _stub_search(
            monkeypatch,
            [[_hit("insight:a1", "A")], [_hit("insight:b1", "B")]],
        )
        compare_subjects(
            tmp_path,
            SubjectRef(kind="person", id="person:alice", label="Alice"),
            SubjectRef(kind="person", id="person:bob", label="Bob"),
            q="",
        )
        assert calls[0]["query"] == "Alice"
        assert calls[1]["query"] == "Bob"

    def test_shared_q_wins_over_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stub_search(
            monkeypatch,
            [[_hit("insight:a1", "A")], [_hit("insight:b1", "B")]],
        )
        compare_subjects(
            tmp_path,
            SubjectRef(kind="person", id="person:alice", label="Alice"),
            SubjectRef(kind="person", id="person:bob", label="Bob"),
            q="AI governance",
        )
        assert calls[0]["query"] == "AI governance"
        assert calls[1]["query"] == "AI governance"

    def test_empty_shared_q_and_empty_label_returns_ungrounded_without_calling_search(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No query material at all → we don't even call the search
        layer for that side; pack comes back as placeholder."""
        calls = _stub_search(
            monkeypatch,
            [[_hit("insight:b1", "B")]],
        )
        outcome = compare_subjects(
            tmp_path,
            SubjectRef(kind="person", id="   ", label=None),
            SubjectRef(kind="person", id="person:bob", label="Bob"),
            q="",
        )
        assert outcome.pack_a.grounded is False
        assert outcome.pack_a.result_count == 0
        # Only subject B triggered a search call.
        assert len(calls) == 1
        assert calls[0]["query"] == "Bob"

    def test_result_count_reflects_returned_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_search(
            monkeypatch,
            [
                [_hit(f"insight:a{i}", f"A{i}") for i in range(3)],
                [_hit("insight:b1", "B1")],
            ],
        )
        outcome = compare_subjects(
            tmp_path,
            SubjectRef(kind="person", id="A"),
            SubjectRef(kind="person", id="B"),
            q="x",
        )
        assert outcome.pack_a.result_count == 3
        assert outcome.pack_b.result_count == 1
