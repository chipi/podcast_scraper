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
