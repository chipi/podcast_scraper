"""The corpus scoreboard counts what a listener can see, by the method's definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.quality.corpus_scoreboard import score_corpus

pytestmark = [pytest.mark.unit]

HOST = "Ann Host: Welcome to the show. "
GUEST = "SPEAKER_01: I spent ten years building bridges and learned patience. "
TAIL = "Ann Host: This episode is brought to you by Acme. "
ADFREE = HOST + GUEST + TAIL


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


def _segments() -> List[Dict[str, Any]]:
    a, b = len(HOST), len(HOST) + len(GUEST)
    return [
        {
            "speaker": "SPEAKER_00",
            "speaker_label": "Ann Host",
            "text": HOST,
            "char_start": 0,
            "char_end": a,
        },
        {"speaker": "SPEAKER_01", "text": GUEST, "char_start": a, "char_end": b},
        {
            "speaker": "SPEAKER_00",
            "speaker_label": "Ann Host",
            "text": TAIL,
            "char_start": b,
            "char_end": len(ADFREE),
        },
    ]


def _quote(qid: str, needle: str, ref: str = "transcripts/ep.adfree.txt") -> Dict[str, Any]:
    start = ADFREE.index(needle)
    return {
        "id": qid,
        "type": "Quote",
        "properties": {
            "text": needle,
            "char_start": start,
            "char_end": start + len(needle),
            "transcript_ref": ref,
        },
    }


def _insight(iid: str, **props: Any) -> Dict[str, Any]:
    base = {"text": f"insight {iid}", "tier": 3, "routing_tag": "surface"}
    base.update(props)
    return {"id": iid, "type": "Insight", "properties": base}


def _corpus(
    root: Path,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    *,
    guest_talk: float,
    feed: str = "f",
    language: str = "en",
) -> None:
    run = root / "feeds" / feed / "run_1"
    _w(
        run / "metadata" / "ep.metadata.json",
        {
            "feed": {"title": f"Show {feed}", "feed_id": feed},
            "episode": {"episode_id": f"ep-{feed}", "language": language},
            "content": {
                "transcript_file_path": f"/app/output/feeds/{feed}/run_1/transcripts/ep.txt"
            },
        },
    )
    _w(run / "metadata" / "ep.gi.json", {"nodes": nodes, "edges": edges})
    _w(run / "transcripts" / "ep.txt", ADFREE)
    _w(run / "transcripts" / "ep.adfree.txt", ADFREE)
    _w(run / "transcripts" / "ep.adfree.segments.json", _segments())
    _w(
        run / "transcripts" / "ep.speakers.diagnostics.json",
        {
            "voices": [
                {"voice": "SPEAKER_00", "talk_time_s": 100.0 - guest_talk},
                {"voice": "SPEAKER_01", "talk_time_s": guest_talk},
            ]
        },
    )


def _sup(insight: str, quote: str) -> Dict[str, str]:
    return {"type": "SUPPORTED_BY", "from": insight, "to": quote}


def test_each_hidden_kind_is_counted_once_by_its_own_definition(tmp_path: Path) -> None:
    nodes = [
        _quote("q1", "Welcome to the show."),
        _quote("q2", "I spent ten years building bridges"),
        _insight("i_visible", speaker="Ann Host"),
        _insight(
            "i_unknown_raw", speaker="SPEAKER_01", speaker_voice_type="unknown", surfaceable=False
        ),
        _insight(
            "i_named_hidden", speaker="Bo Guest", speaker_voice_type="unknown", surfaceable=False
        ),
        _insight(
            "i_unid", speaker="SPEAKER_01", speaker_voice_type="unidentified", surfaceable=False
        ),
        _insight("i_contradiction", speaker="Ann Host", surfaceable=False),
    ]
    edges = [_sup("i_visible", "q1"), _sup("i_unid", "q2")]
    _corpus(tmp_path, nodes, edges, guest_talk=40.0)
    t = score_corpus(tmp_path)["total"]
    assert t["insights"] == 5
    assert t["S1_hidden_unknown"] == 2
    assert t["S1n_named_but_hidden"] == 1  # the real name that is still hidden: 0031's failure
    assert t["S1c_inconsistent"] == 1  # no type, yet hidden
    assert t["S2_hidden"] == 4
    assert t["S2_hidden_unidentified"] == 1
    assert t["S3_principal_unidentified"] == 1  # the guest holds 40% >= 30%
    assert t.get("S12_quotes_off_text", 0) == 0
    assert t["A1_marker_hits"] == 1 and t["A1_episodes_with_hits"] == 1


def test_s3_finds_the_voice_through_the_quote_not_the_renamed_speaker(tmp_path: Path) -> None:
    """After a rename the insight's ``speaker`` is a name; the voice is the quote's segment."""
    nodes = [
        _quote("q2", "I spent ten years building bridges"),
        _insight(
            "i_unid", speaker="Bo Guest", speaker_voice_type="unidentified", surfaceable=False
        ),
    ]
    _corpus(tmp_path, nodes, [_sup("i_unid", "q2")], guest_talk=40.0)
    assert score_corpus(tmp_path)["total"]["S3_principal_unidentified"] == 1
    _corpus(tmp_path, nodes, [_sup("i_unid", "q2")], guest_talk=10.0)
    assert score_corpus(tmp_path)["total"].get("S3_principal_unidentified", 0) == 0


def test_a_quote_off_its_text_and_a_missing_transcript_are_told_apart(tmp_path: Path) -> None:
    off = _quote("q1", "Welcome to the show.")
    off["properties"]["char_start"] += 3
    off["properties"]["char_end"] += 3
    gone = _quote("q2", "Welcome to the show.", ref="transcripts/missing.adfree.txt")
    _corpus(tmp_path, [off, gone, _insight("i", speaker="Ann Host")], [], guest_talk=40.0)
    t = score_corpus(tmp_path)["total"]
    assert t["S12_quotes_off_text"] == 1
    assert t["quotes_ref_unresolved"] == 1


def test_an_episode_whose_insights_are_all_hidden_shows_none_but_its_panel_falls_back(
    tmp_path: Path,
) -> None:
    nodes = [
        _quote("q2", "I spent ten years building bridges"),
        _insight(
            "i_unknown", speaker="SPEAKER_01", speaker_voice_type="unknown", surfaceable=False
        ),
    ]
    _corpus(tmp_path, nodes, [_sup("i_unknown", "q2")], guest_talk=40.0)
    t = score_corpus(tmp_path)["total"]
    assert t["S10a_zero_visible"] == 1
    assert t.get("S10b_zero_visible_panel", 0) == 0


def test_counts_are_also_reported_per_language(tmp_path: Path) -> None:
    _corpus(tmp_path, [_insight("i", speaker="Ann Host")], [], guest_talk=40.0)
    board = score_corpus(tmp_path)
    assert board["per_language"]["en"]["insights"] == 1
    assert board["total"]["insights"] == 1


def test_a_named_voice_still_typed_unidentified_is_named_but_hidden(tmp_path: Path) -> None:
    """The 526 on the 10-16 snapshot: a real name, type ``unidentified``, hidden."""
    nodes = [
        _insight("i", speaker="Kent Beck", speaker_voice_type="unidentified", surfaceable=False),
    ]
    _corpus(tmp_path, nodes, [], guest_talk=40.0)
    assert score_corpus(tmp_path)["total"]["S1n_named_but_hidden"] == 1


def test_s3_uses_the_first_grounding_quote_not_a_later_one(tmp_path: Path) -> None:
    nodes = [
        _quote("q_guest", "I spent ten years building bridges"),
        _quote("q_host", "Welcome to the show."),
        _insight("i", speaker="SPEAKER_01", speaker_voice_type="unidentified", surfaceable=False),
    ]
    edges = [_sup("i", "q_guest"), _sup("i", "q_host")]
    _corpus(tmp_path, nodes, edges, guest_talk=75.0)  # the host holds 25%, under the threshold
    assert score_corpus(tmp_path)["total"]["S3_principal_unidentified"] == 1


def test_a_quote_starting_on_a_segment_boundary_belongs_to_that_segment(tmp_path: Path) -> None:
    nodes = [
        _quote("q", "SPEAKER_01: I spent"),  # starts exactly at the guest segment's char_start
        _insight("i", speaker="SPEAKER_01", speaker_voice_type="unidentified", surfaceable=False),
    ]
    _corpus(tmp_path, nodes, [_sup("i", "q")], guest_talk=40.0)
    t = score_corpus(tmp_path)["total"]
    assert t["S3_principal_unidentified"] == 1 and t["S3_outside_no_voice"] == 0


def test_each_language_gets_its_own_row_and_zeros_are_written_out(tmp_path: Path) -> None:
    _corpus(tmp_path, [_insight("i", speaker="Ann Host")], [], guest_talk=40.0, feed="a")
    _corpus(
        tmp_path, [_insight("j", speaker="Ann Host")], [], guest_talk=40.0, feed="b", language="de"
    )
    board = score_corpus(tmp_path)
    assert board["per_language"]["en"]["insights"] == 1
    assert board["per_language"]["de"]["insights"] == 1
    assert board["total"]["insights"] == 2
    assert board["total"]["S1n_named_but_hidden"] == 0  # present, not missing
    assert board["total"]["episodes_no_gi"] == 0


def test_talk_that_mentions_code_or_a_sponsored_bill_is_not_an_ad_marker(tmp_path: Path) -> None:
    from podcast_scraper.quality.corpus_scoreboard import _AD_MARKER

    talk = "We use Codex because code is cheap. The bill was sponsored by two senators."
    assert _AD_MARKER.findall(talk) == []
    ads = "This episode is brought to you by Acme. Use promo code PODS. We are also sponsored by X."
    assert len(_AD_MARKER.findall(ads)) == 3
