"""An advertisement is never evidence, and an unattributed stance is not a stance.

The diarization roster types every voice — person / cameo / commercial / unknown / unidentified —
and GI ignored all of it. So:

* ad copy could be grounded as an insight. Ad copy is *written* to be quotable ("if you play our
  games, you probably know there's something a bit different about them"), which makes it the most
  fluent, most confident false insight available;
* an anonymous voice could mint a Person node called ``SPEAKER_09``, which a downstream enricher
  then had to filter back out (#1167) — a mop, not a gate.

Two rules, and they are NOT the same rule:

    commercial     never grounded. There is no insight in an advertisement.
    unknown        a person we FAILED to name — not surfaceable, and COUNTED, because a defect that
                   costs nothing gets fixed by nobody.
    unidentified   a person NOBODY names (the vox-pop of a narrated piece) — not surfaceable, but
                   still eligible for CONNECT.

That last distinction is the whole design. On Planet Money and The Daily the tape IS the story —
36-40% of those episodes — so discarding unidentified speech outright would gut the narrated shows
to protect them from a problem they do not have. A fact is still a fact; only a STANCE needs a name.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.gi.grounding import GroundedQuote
from podcast_scraper.gi.pipeline import build_artifact

AD_LINE = "If you play our games, you probably know there is something a bit different about them."
TAPE_LINE = "It was one of the greatest things that ever happened to me in my life, honestly."
HOST_LINE = "The airline shut down this week and every flight was cancelled immediately."


def _segments() -> List[Dict[str, Any]]:
    """A narrated episode: an advert, a named host, and a vox-pop nobody names."""
    segs = [
        {
            "start": 0.0,
            "end": 8.0,
            "text": AD_LINE,
            "speaker": "SPEAKER_04",
            "speaker_label": "SPEAKER_04",
            "voice_type": "commercial",
        },
        {
            "start": 8.0,
            "end": 20.0,
            "text": HOST_LINE,
            "speaker": "SPEAKER_00",
            "speaker_label": "Alexi Horowitz-Gazi",
        },
        {
            "start": 20.0,
            "end": 40.0,
            "text": TAPE_LINE,
            "speaker": "SPEAKER_09",
            "speaker_label": "SPEAKER_09",
            "voice_type": "unidentified",
        },
    ]
    cursor = 0
    for s in segs:
        text = str(s["text"])
        s["char_start"] = cursor
        s["char_end"] = cursor + len(text)
        cursor = cursor + len(text) + 1
    return segs


def _transcript(segs: List[Dict[str, Any]]) -> str:
    return " ".join(s["text"] for s in segs)


def _cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.generate_gi = True
    cfg.gi_require_grounding = True
    cfg.gi_fail_on_missing_grounding = False
    cfg.extractive_qa_device = None
    cfg.nli_device = None
    return cfg


def _artifact_for(line: str) -> Dict[str, Any]:
    segs = _segments()
    text = _transcript(segs)
    seg = next(s for s in segs if s["text"] == line)
    grounded = [
        GroundedQuote(
            char_start=seg["char_start"],
            char_end=seg["char_end"],
            text=line,
            qa_score=0.9,
            nli_score=0.8,
        )
    ]
    with (
        patch(
            "podcast_scraper.gi.deps.create_gil_evidence_providers",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "podcast_scraper.gi.grounding.find_grounded_quotes_via_providers",
            return_value=grounded,
        ),
    ):
        return build_artifact(
            "episode:pm-0001",
            text,
            cfg=_cfg(),
            transcript_segments=segs,
            transcript_ref="transcripts/0001 - ep.txt",
        )


def _quotes(artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [n for n in artifact["nodes"] if n["type"] == "Quote"]


def _insights(artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [n for n in artifact["nodes"] if n["type"] == "Insight"]


def test_an_advertisement_is_never_grounded() -> None:
    """THE FALSE INSIGHT. Ad copy is engineered to be quotable — refuse it at the Quote."""
    artifact = _artifact_for(AD_LINE)
    quotes = _quotes(artifact)
    assert not quotes, (
        f"a span inside an ADVERTISEMENT was grounded as evidence: "
        f"{[q['properties']['text'] for q in quotes]}"
    )
    for n in artifact["nodes"]:
        assert n["type"] != "Person", "an advert must not mint a Person node"


def test_a_named_person_is_surfaceable() -> None:
    artifact = _artifact_for(HOST_LINE)
    assert _quotes(artifact), "a named person's words must ground normally"
    ins = _insights(artifact)[0]["properties"]
    assert ins.get("surfaceable") is not False
    assert ins.get("speaker_voice_type") in (None, "person")


class TestTheTapeOfANarratedShow:
    """`unidentified` — a real person, real testimony, and nobody in the episode names them."""

    def test_it_is_still_grounded(self) -> None:
        """We do NOT discard it. On Planet Money and The Daily the tape IS the story."""
        artifact = _artifact_for(TAPE_LINE)
        assert _quotes(artifact), (
            "unidentified speech was dropped entirely — that guts the narrated shows, where the "
            "vox-pop carries 36-40% of the episode"
        )

    def test_but_it_is_not_surfaceable(self) -> None:
        """An unattributed STANCE is not a stance — nobody holds it, nobody can disagree with it."""
        ins = _insights(_artifact_for(TAPE_LINE))[0]["properties"]
        assert ins["speaker_voice_type"] == "unidentified"
        assert ins["surfaceable"] is False


def test_the_exclusions_are_logged_not_silent(caplog: pytest.LogCaptureFixture) -> None:
    """A silent exclusion is a cost nobody can see."""
    with caplog.at_level("INFO"):
        _artifact_for(AD_LINE)
    assert "advertisement" in caplog.text.lower()


class TestThePersonNodeMustBeAPerson:
    """19% of the Person nodes in the shipped corpus were called "SPEAKER_NN".

    GI minted a Person for every unresolved voice and hung a SPOKEN_BY edge on it. #1167 then
    filtered them back out of the trending/consensus surfaces — a mop, not a gate, and one that only
    worked because the id happened to be ugly. Give those voices a friendly name and it breaks.

    The roster already knows the voice is not a person. So the graph must not be told otherwise.
    """

    def test_an_unidentified_voice_mints_no_person(self) -> None:
        artifact = _artifact_for(TAPE_LINE)
        people = [n for n in artifact["nodes"] if n["type"] == "Person"]
        assert not people, (
            "a voice nobody names became a Person in the graph: "
            f"{[(p.get('properties') or {}).get('name') for p in people]}"
        )
        assert not [e for e in artifact["edges"] if e["type"] == "SPOKEN_BY"]

    def test_the_quote_still_says_who_spoke(self) -> None:
        """The surface must be able to name the speaker WITHOUT the graph inventing one."""
        q = _quotes(_artifact_for(TAPE_LINE))[0]["properties"]
        assert q["speaker_name"] == "Unidentified speaker"
        assert q["speaker_voice_type"] == "unidentified"
        assert q["speaker_id"] is None

    def test_a_real_person_still_gets_a_person_node(self) -> None:
        artifact = _artifact_for(HOST_LINE)
        names = {
            (n.get("properties") or {}).get("name")
            for n in artifact["nodes"]
            if n["type"] == "Person"
        }
        assert "Alexi Horowitz-Gazi" in names
        assert [e for e in artifact["edges"] if e["type"] == "SPOKEN_BY"]


def test_an_unsurfaceable_insight_never_reaches_the_ui() -> None:
    """`surfaceable` was written by GI and read by NOBODY — a gate wired to nothing.

    `insights_from_gi` is the surfacing point. An unattributed stance must not come out of it,
    however the classifier labelled it.
    """
    from podcast_scraper.server.app_gi_view import insights_from_gi

    tape = _artifact_for(TAPE_LINE)
    assert _insights(tape), "the insight must still EXIST — the corpus needs it for CONNECT"
    assert (
        insights_from_gi(tape) == []
    ), "an insight spoken by a voice nobody can name was published to the UI as somebody's insight"

    host = _artifact_for(HOST_LINE)
    assert insights_from_gi(host), "a named person's insight must still surface"
