"""A quote whose transcript was rewritten AFTER GI was built is re-anchored, not dropped.

A speaker relabel rewrites every turn label ("SPEAKER_01:" -> "Tracy Alloway:"), so every
character after the first changed label moves while the quote text itself is untouched. Measured on
prod 2026-10-02: 36 relabelled episodes, 2,021 quotes skipped on every enrichment run, 1,997 of them
occurring exactly once in the current transcript.

Each case below is a separate shape: the moved quote, the quote whose text occurs twice, the quote
whose text is gone, and the quote that never moved.

All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

from typing import Dict, List

from podcast_scraper.gi.speakers import add_spoken_by_edges

HOST = "Tobias Wren"
GUEST = "Maria Lindqvist"

# Enough relabelled turns before the quotes that the shift exceeds the probe's 64-char slack (the
# measured prod shifts were 1,400-2,400 chars).
_PREAMBLE = "".join(
    f"SPEAKER_0{i % 2}: Short line number {i} before the conversation proper.\n" for i in range(40)
)
OLD = _PREAMBLE + (
    "SPEAKER_00: Welcome back to the show.\n"
    "SPEAKER_01: The ports moved north because the river silted up.\n"
    "SPEAKER_00: And the merchants followed?\n"
    "SPEAKER_01: Most of them, within a generation.\n"
)
NEW = OLD.replace("SPEAKER_00", HOST).replace("SPEAKER_01", GUEST)


def _quote(qid: str, text: str, transcript: str) -> Dict:
    start = transcript.index(text)
    return {
        "id": qid,
        "type": "Quote",
        "properties": {"text": text, "char_start": start, "char_end": start + len(text)},
    }


def _artifact(quotes: List[Dict]) -> Dict:
    return {"episode_id": "ep-1", "nodes": quotes, "edges": []}


def _props(artifact: Dict, qid: str) -> Dict:
    return next(n["properties"] for n in artifact["nodes"] if n["id"] == qid)


def _spoken_by(artifact: Dict, qid: str) -> List[str]:
    return [e["to"] for e in artifact["edges"] if e["type"] == "SPOKEN_BY" and e["from"] == qid]


def test_a_moved_quote_gets_the_new_offsets_and_its_speaker() -> None:
    text = "The ports moved north because the river silted up."
    art = _artifact([_quote("q1", text, OLD)])
    add_spoken_by_edges(art, NEW, hosts=[HOST], guests=[GUEST])
    p = _props(art, "q1")
    assert NEW[p["char_start"] : p["char_end"]] == text
    assert len(_spoken_by(art, "q1")) == 1


def test_a_moved_quote_whose_text_occurs_twice_is_left_unattributed() -> None:
    # Either occurrence could be the one the model quoted; guessing would credit the wrong voice.
    old = OLD + "SPEAKER_00: Most of them, within a generation.\n"
    new = old.replace("SPEAKER_00", HOST).replace("SPEAKER_01", GUEST)
    text = "Most of them, within a generation."
    art = _artifact([_quote("q1", text, old)])
    before = dict(_props(art, "q1"))
    add_spoken_by_edges(art, new, hosts=[HOST], guests=[GUEST])
    assert _props(art, "q1") == before
    assert _spoken_by(art, "q1") == []


def test_a_quote_whose_text_is_gone_is_left_unattributed() -> None:
    art = _artifact([_quote("q1", "The ports moved north because the river silted up.", OLD)])
    rewritten = NEW.replace("silted up", "dried out")
    before = dict(_props(art, "q1"))
    add_spoken_by_edges(art, rewritten, hosts=[HOST], guests=[GUEST])
    assert _props(art, "q1") == before
    assert _spoken_by(art, "q1") == []


def test_an_aligned_quote_keeps_its_offsets() -> None:
    text = "The ports moved north because the river silted up."
    art = _artifact([_quote("q1", text, NEW)])
    before = dict(_props(art, "q1"))
    add_spoken_by_edges(art, NEW, hosts=[HOST], guests=[GUEST])
    assert _props(art, "q1") == before
    assert len(_spoken_by(art, "q1")) == 1
