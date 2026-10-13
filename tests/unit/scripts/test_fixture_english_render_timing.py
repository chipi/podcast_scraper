"""The translated shows' English renders run on the episode's own timeline (operator 2026-10-12).

A short listen cut its clip from the English render at the quote's time, but the captured render
carried another run's timings, so "Start at strava.com/podcast" played for a point about drainage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

import build_app_validation_corpus as builder  # noqa: E402

pytestmark = [pytest.mark.unit]

TRANSCRIPTS = (
    ROOT / "tests/fixtures/app-validation-corpus/v3/feeds/p10/run_20260101-000000/transcripts"
)


def _segs(name: str) -> list[dict]:
    segs: list[dict] = builder._segments_of(json.loads((TRANSCRIPTS / name).read_text("utf-8")))
    return segs


def test_every_english_sentence_sits_inside_a_source_segment() -> None:
    source = _segs("p10_e01.es.segments.json")
    for seg in _segs("p10_e01.segments.json"):
        assert any(
            s["start"] <= seg["start"] and seg["end"] <= s["end"] + 0.01 for s in source
        ), seg["text"]


def test_the_point_about_drainage_plays_where_its_quote_is() -> None:
    drainage = next(s for s in _segs("p10_e01.segments.json") if s["text"].startswith("Honestly"))
    assert drainage["start"] == 24.0  # the GI quote's 24-30 s, not the ad's 23-25 s


def test_the_committed_renders_need_no_retiming() -> None:
    assert builder.retime_english_render(TRANSCRIPTS, "p10_e01", apply=False) == 0


def test_an_episode_without_a_translation_has_no_render_to_retime(tmp_path: Path) -> None:
    assert builder.english_render_times(tmp_path, "p01_e01") == {}


@pytest.mark.parametrize(
    "line",
    [
        "We'll be right back after a quick word from our sponsors.",
        "Volvemos enseguida, después de una breve pausa publicitaria.",
        "Torniamo subito, dopo una breve pausa pubblicitaria.",
        "Nous revenons tout de suite, après une courte pause publicitaire.",
        "Wir sind gleich zurück, nach einer kurzen Werbepause.",
        "Voltamos já a seguir, depois de uma breve pausa publicitária.",
    ],
)
def test_the_break_lead_in_is_ad_copy_in_every_language(line: str) -> None:
    assert builder.is_ad_copy(line)


def test_a_point_is_not_ad_copy() -> None:
    assert not builder.is_ad_copy("Honestly, drainage is the design decision with the most impact.")


# --- The claim test sees every translated line's English (2026-10-12) ------------------------
#
# The translated shows' later episodes kept only connective templates as insights: their English
# was known only for an opening stretch, and the translations paraphrase the English templates.


@pytest.mark.parametrize(
    "english",
    [
        "Before we get started, Javier Benavides was on the show discussing how drainage matters.",
        "Before we get into the details, Julien Mercier was on the show and explained that.",
        "Yes, exactly. And that ties in with what we're discussing today.",
        "Yes, exactly. And it connects to what we're going to be discussing today.",
        "I want to point something out – Marta Solís mentioned this same dynamic.",
        "I'd like to point something out – Stefan Brandt mentioned a similar dynamic.",
        "Let's change the subject – I want to dedicate the final part to land management.",
        "That's all for today's episode of Path Conversations. Until next time.",
        "Before we close, thank you again to our friends at Linear.",
        "Sophie Lorenz, thanks for the conversation.",
        "Julien Mercier, thank you for this conversation.",
        "That's it for this Long View. Until next time.",
    ],
)
def test_a_connective_template_is_not_a_claim_in_any_wording(english: str) -> None:
    sys.path.insert(0, str(ROOT / "scripts" / "tools"))
    import realign_fixture_gi as realign  # noqa: E402

    assert realign.is_not_a_claim("(source-language line)", english)


def test_a_point_is_still_a_claim() -> None:
    sys.path.insert(0, str(ROOT / "scripts" / "tools"))
    import realign_fixture_gi as realign  # noqa: E402

    point = "Honestly, drainage is the design decision that has the biggest impact on any trail."
    assert not realign.is_not_a_claim("Sinceramente, el drenaje es la decisión.", point)


def test_a_translated_episode_is_grounded_in_its_english_turns() -> None:
    """D-44: a translated episode's GI is built on its English ad-free body, so the realign grounds
    it in the same English turns the builder used (not the source lines it paired with English
    before D-44 reached this corpus), on the source timeline, indexing ``<stem>.adfree.txt``."""
    sys.path.insert(0, str(ROOT / "scripts" / "tools"))
    import realign_fixture_gi as realign  # noqa: E402

    gi = TRANSCRIPTS.parent / "metadata" / "p10_e02.gi.json"
    segments, body, turns, english = realign._episode_inputs(gi)
    assert turns is None and english is None, "English insights need no pairing"
    assert body == (TRANSCRIPTS / "p10_e02.adfree.txt").read_text("utf-8")
    assert segments and all(body[s["char_start"] : s["char_end"]] == s["text"] for s in segments)
    source = json.loads((TRANSCRIPTS / "p10_e02.es.segments.json").read_text("utf-8"))
    end = max(float(s["end"]) for s in source)
    assert all(0.0 <= float(s["start"]) <= float(s["end"]) <= end + 0.01 for s in segments)
