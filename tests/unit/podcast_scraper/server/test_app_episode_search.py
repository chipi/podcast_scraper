"""What a listener remembers, found in the episode's timed transcript (operator 2026-10-10)."""

from __future__ import annotations

from types import SimpleNamespace

from podcast_scraper.server.app_episode_search import exact_passages, time_transcript_hits

SEGS = [
    (0, 4_000, "Welcome back to the show."),
    (4_000, 9_000, "Today we talk about sizing positions so the"),
    (9_000, 14_000, "worst week is survivable for a long time."),
    (14_000, 20_000, "Diversification is the only free lunch, people say."),
    (20_000, 26_000, "But the worst week can still arrive in any portfolio."),
]


def _exact(query: str) -> list[dict]:
    return exact_passages(SEGS, query, episode_id="ep1")


def test_a_remembered_phrase_finds_its_passage_with_its_real_time():
    hits = _exact("worst week is survivable")
    assert hits[0]["metadata"]["timestamp_start_ms"] == 9_000
    assert hits[0]["metadata"]["match"] == "phrase"
    assert hits[0]["metadata"]["doc_type"] == "transcript"


def test_a_phrase_cut_across_two_segments_is_still_found():
    # "so the worst week" spans the boundary between segments 1 and 2.
    hits = _exact("so the worst week")
    assert hits[0]["metadata"]["timestamp_start_ms"] == 4_000
    assert hits[0]["metadata"]["timestamp_end_ms"] == 14_000
    assert hits[0]["metadata"]["match"] == "phrase"


def test_words_in_any_order_come_after_the_exact_phrase():
    hits = _exact("worst week")
    assert [h["metadata"]["timestamp_start_ms"] for h in hits] == [9_000, 20_000]
    assert [h["metadata"]["match"] for h in hits] == ["phrase", "phrase"]
    reordered = _exact("week worst")
    assert {h["metadata"]["match"] for h in reordered} == {"words"}


def test_case_and_filler_words_do_not_matter_but_whole_words_do():
    assert _exact("DIVERSIFICATION")[0]["metadata"]["timestamp_start_ms"] == 14_000
    assert _exact("the free lunch")[0]["metadata"]["timestamp_start_ms"] == 14_000
    assert _exact("diversify") == []  # whole words only: another form is meaning search's job


def test_nothing_when_the_words_are_not_there():
    assert _exact("inflation") == []
    assert _exact("   ") == []


def _hit(text: str, start=0, doc_type="transcript"):
    return SimpleNamespace(
        text=text,
        metadata={"doc_type": doc_type, "timestamp_start_ms": start, "timestamp_end_ms": 0},
    )


def test_a_transcript_result_gets_the_time_its_words_were_said():
    hit = _hit("we talk about sizing positions so the worst week is survivable for a long time")
    time_transcript_hits([hit], SEGS)
    assert hit.metadata["timestamp_start_ms"] == 4_000
    assert hit.metadata["timestamp_end_ms"] == 14_000


def test_speaker_labels_in_the_result_do_not_stop_it_being_found():
    # The prod chunk text carries "Name:" lines that the timed segments do not.
    hit = _hit("Nora: Diversification is the only free lunch, people say. But the worst week")
    time_transcript_hits([hit], SEGS)
    assert hit.metadata["timestamp_start_ms"] == 14_000


def test_a_result_that_cannot_be_found_loses_its_false_0_00():
    hit = _hit("completely different words that were never said in this episode at all")
    time_transcript_hits([hit], SEGS)
    assert hit.metadata["timestamp_start_ms"] is None
    assert hit.metadata["timestamp_end_ms"] is None


def test_other_results_and_exact_passages_are_left_alone():
    insight = _hit("an insight about risk", start=12_345, doc_type="insight")
    exact = SimpleNamespace(
        text="x",
        metadata={"doc_type": "transcript", "match": "phrase", "timestamp_start_ms": 9_000},
    )
    time_transcript_hits([insight, exact], SEGS)
    assert insight.metadata["timestamp_start_ms"] == 12_345
    assert exact.metadata["timestamp_start_ms"] == 9_000
